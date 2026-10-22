import os
import torch
import torch.autograd.forward_ad as fwAD

import time
from datetime import timedelta
from torch import nn
from typing import Optional
from torch.amp import autocast

from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from .utils import get_inr_param_count_from_shapes, unflatten_to_wsfeat, SurrogateAttacker
from utils.wandb_manager import WandbManager
from basemodels.inr2vec.pcd_retrieval_and_classification.pcd import compute_udf_from_pcd, farthest_point_sampling, random_point_sampling, shuffle_pcd
from basemodels.inr2vec.utils import mlp_batched_forward


class SimpleTimer:
    def __init__(self, total_steps, start_step=0):
        self.total = total_steps
        self.start_step = start_step
        self.start = time.time()

    def get_eta(self, current_step):
        elapsed = time.time() - self.start
        steps_done = current_step - self.start_step
        speed = steps_done / elapsed if elapsed > 0 and steps_done > 0 else 0
        eta = (self.total - current_step) / speed if speed > 0 else 0
        return str(timedelta(seconds=int(eta)))


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class SurrogateTrainer:
    def __init__(self, 
                 dataset_name, 
                 victim_wrapped, 
                 builder, 
                 dataloader, 
                 exp_ckpt_path, 
                 lam_inr=1.0, lam_der=1.0, lam_cls=1.0, lam_recon=0.0, lam_emb=1.0, alpha=0.6/7, epochs=100, lr=5e-4,
                 device='cuda',
                 patience=10,
                 wandb_manager=None,
                 log_image_interval=100,
                 type='FD',
                 pgd_aug=False,
                 n_aug=0,
                 **kwargs,):
        
        self.dataset_name = dataset_name
        self.victim = victim_wrapped
        self.builder = builder
        self.trainloader, self.valloader, self.testloader = dataloader
        print(f"Trainloader length: {len(self.trainloader)}, Valloader length: {len(self.valloader)}, Testloader length: {len(self.testloader)}")
        self.exp_ckpt_path = exp_ckpt_path
        self.lam_inr = lam_inr
        self.lam_der = lam_der
        self.lam_cls = lam_cls
        self.lam_recon = lam_recon
        self.lam_emb = lam_emb
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.type = type
        self.patience = patience
        self.early_stop_from = kwargs.get('early_stop_from', 0)
        self.wandb_manager = wandb_manager
        self.device = device
        self.is_FD = type == 'FD'
        self.is_i2i = kwargs['surrogate_type'].startswith("i2i")
        self.is_3d = self.dataset_name == "shapenet"
        self.verbose = kwargs.get('verbose', False)
        self.use_precomputed_embs = kwargs.get('use_precomputed_embs', False)
        self.resume_epoch = 0
        self.exp_plot_path = kwargs['exp_plot_path']
        self.n_aug = n_aug
        self.test_only = kwargs.get('test_only', False)
        self.amp = kwargs.get('amp', False)
        self.bigger_surr = kwargs.get('bigger_surr', False)
        self.kwargs = kwargs

        if self.n_aug > 0:
            from attacks.run_attacks import DEFAULT_EPS
            self.aug_eps = DEFAULT_EPS[self.dataset_name]  # image-space noise augmentation for OOD robustness
            print(f"Using image augmentation with eps={self.aug_eps} and n_aug={self.n_aug}")

        self.pgd_aug = pgd_aug
        self.init_surrogate()
        
        if self.pgd_aug:
            from attacks.attack_methods import get_attacker
            print(f"[Surrogate Training] Using PGD augmentation with norm={kwargs.get('norm', 'linf')}")
            self.surr_attacker = SurrogateAttacker(is_i2i=self.is_i2i, surrogate=self.surrogate, victim=self.victim, weight_shapes=self.weight_shapes, bias_shapes=self.bias_shapes, is_3d=self.is_3d)
            self.attack = get_attacker(self.surr_attacker, self.dataset_name, kwargs={"steps": kwargs.get("surr_train_steps", 5), "norm": kwargs.get("norm", "linf"), "attack_space": "image", "attack_type":kwargs['attack_type']})

        self.log_image_interval = log_image_interval if self.is_i2i else 0
        self.step = self.step_i2i if self.is_i2i else self.step_i2e
        
        if self.is_3d:
            self.num_queries_on_surface = kwargs.get('num_queries_on_surface', 1000)
            self.stds = kwargs.get('stds', [0.003, 0.01, 0.05, 0.1])
            self.num_points_per_std = kwargs.get('num_points_per_std', [1000, 500, 500, 500])

    def init_surrogate(self):
        c_out = 3 if self.dataset_name == "cifar" else 1
        size = 32 if self.dataset_name == "cifar" else 28

        if self.is_i2i:
            sample_batch = next(iter(self.trainloader))
            sample_wsfeat = WeightSpaceFeatures(*sample_batch[0]).to("cpu")
            self.weight_shapes = [tuple(w.shape[1:]) for w in sample_wsfeat.weights]
            self.bias_shapes = [tuple(b.shape[1:]) for b in sample_wsfeat.biases]
            total_params = get_inr_param_count_from_shapes(self.weight_shapes, self.bias_shapes)
            print(f"Total INR params per sample: {total_params}")
            out_features = total_params
            self.victim.eval()
        else: 
            self.weight_shapes, self.bias_shapes = None, None
            out_features = self.victim.get_embedding_dim()

        self.surrogate = MLP(
            in_features=2048 * 3 if self.is_3d else c_out * size * size,
            out_features=out_features,
            hidden_features=self.kwargs.get("surrogate_hidden_features", 512),
            hidden_layers=self.kwargs.get("surrogate_hidden_layers", 3),
        ).cuda()

    def load_surrogate(self, path):
        self.surrogate.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
        self.resume_epoch = self.kwargs.get('resume_epoch', 0)
        print(f"Resumed surrogate training from {path}. Starting at epoch {self.resume_epoch}")

    def evaluate_inr_on_coords(self, wsfeat, coords):
        """Evaluate INR (from wsfeat) on query coordinates using siren_batched_forward.
        
        Args:
            wsfeat: WeightSpaceFeatures
            coords: (B, N, 3) query points
        Returns:
            (B, N) UDF predictions (logits)
        """
        # Convert wsfeat to [w0, b0, w1, b1, ...] format
        batched_params = []
        for w, b in zip(wsfeat.weights, wsfeat.biases):
            # Remove channel dim if present: (B, 1, H, W) -> (B, H, W)
            if w.dim() == 4 and w.shape[1] == 1:
                w = w.squeeze(1)
            if b.dim() == 3 and b.shape[1] == 1:
                b = b.squeeze(1)
            batched_params.append(w)
            batched_params.append(b)
        
        return mlp_batched_forward(batched_params, coords)

    def flatten_wsfeat(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        """Flatten WeightSpaceFeatures to (batch, total_params) tensor."""
        batch_size = wsfeat.weights[0].shape[0]
        parts = []
        for w in wsfeat.weights:
            parts.append(w.reshape(batch_size, -1))
        for b in wsfeat.biases:
            parts.append(b.reshape(batch_size, -1))
        return torch.cat(parts, dim=1)


    def perturb_weight(self, weight, alpha=0.001):
        """Perturb weight with uniform noise in [-alpha, alpha]."""
        noise = 2 * alpha * torch.rand_like(weight) - alpha
        return weight + noise


    def step_i2e(self, batch, celoss, device='cuda', eval=False):
        params, init_params, src, embs, labels = batch

        device = torch.device("cuda")
        labels = labels.to(device)
        src = src.to(device)   
        # embs = embs.to(device)   
        params = tuple([[i.to(device) for i in inr] for inr in params])
        batch_size = labels.shape[0]

        # Detach to prevent unnecessary gradient computation through the frozen victim encoder.
        if not self.use_precomputed_embs:
            embs = self.victim.encode(WeightSpaceFeatures(*params)).detach()
        flat_embs = embs.view(batch_size, -1)
        if self.is_3d:
            pcd = farthest_point_sampling(src, 2048)
            flat_x = pcd.reshape(batch_size, -1)  # (3072,)
        else:
            flat_x = src.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        pred_embs = self.surrogate(flat_x)                       # (B, E)
        
        loss_emb = (pred_embs - flat_embs).pow(2).mean()
        y_pred = self.victim.classify_embedding(pred_embs)

        loss_cls = celoss(y_pred, labels)

        loss = self.lam_emb * loss_emb + self.lam_cls * loss_cls

        if self.is_FD:
            # Weight-space perturbation
            params = WeightSpaceFeatures(*params).to(device)
            perturbed_params = params.map(lambda w: self.perturb_weight(w, alpha=self.alpha))
            perturbed_params = perturbed_params.to(device) # \theta'

            with torch.no_grad():
                perturbed_embs = self.victim.encode(perturbed_params)  # h' (B, E)

            if self.is_3d:
                delta_emb = ((perturbed_embs - flat_embs) / self.alpha).detach()
                flat_delta_x = torch.randn_like(flat_x)
            else:
                # Render BOTH original and perturbed INR to correctly compute delta_x.
                # Using imgs directly as the baseline introduces SIREN reconstruction error
                # (|render(θ*) - imgs| / alpha), which gets amplified by 1/alpha.
                original_imgs_recon = self.builder.build_img_from_wts_and_bs((params.weights, params.biases))
                perturbed_imgs_recon = self.builder.build_img_from_wts_and_bs((perturbed_params.weights, perturbed_params.biases))
                x_orig_recon = torch.permute(original_imgs_recon, (0, 3, 1, 2)).to(device) * 2.0 - 1.0
                x_p = torch.permute(perturbed_imgs_recon, (0, 3, 1, 2)).to(device) * 2.0 - 1.0 # x'
                    
                delta_x = ((x_p - x_orig_recon) / self.alpha).detach()              # (B,C,H,W)
                delta_emb = ((perturbed_embs - flat_embs) / self.alpha).detach()   # (B, E) 

                flat_delta_x = delta_x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
            
            with fwAD.dual_level():
                dual_input = fwAD.make_dual(flat_x, flat_delta_x)
                _, d_hat = fwAD.unpack_dual(self.surrogate(dual_input))  # (B, E)

            loss_der = (d_hat - delta_emb).pow(2).mean()

            # OOD robustness: also match Jacobian at noise-augmented inputs.
            if not eval:
                if (self.n_aug > 0 and self.aug_eps > 0.0) or self.pgd_aug:
                    if self.pgd_aug:
                        src_aug = self.attack(src, labels)
                        self.victim.eval()
                    else:
                        noise = torch.empty(self.n_aug, *src.shape, device=src.device).uniform_(-self.aug_eps, self.aug_eps)
                        src_aug = (src.unsqueeze(0) + noise).clamp(-1.0, 1.0).flatten(0, 1)  # (n_aug*B, C, H, W)

                    n = 1 if self.pgd_aug else self.n_aug

                    if self.is_3d:
                        flat_x_aug = farthest_point_sampling(src_aug, 2048).reshape(-1, flat_x.shape[-1])
                    else:
                        flat_x_aug = src_aug.permute(0, 2, 3, 1).contiguous().view(-1, flat_x.shape[-1])

                    with fwAD.dual_level():
                        dual_aug = fwAD.make_dual(flat_x_aug, flat_delta_x.repeat(n, 1))
                        _, d_hat_aug = fwAD.unpack_dual(self.surrogate(dual_aug))
                    aug_loss = (d_hat_aug - delta_emb.repeat(n, 1)).pow(2).mean()
                    loss_der = (loss_der + aug_loss) * 0.5

            loss = loss + self.lam_der * loss_der

        metrics = {
            "loss": float(loss.item()),
            "emb": float(loss_emb.item()),
            "cls": float(loss_cls.item()),
            "der": float(loss_der.item()) if self.is_FD else 0.0,
            "acc": float((y_pred.argmax(dim=1) == labels).float().mean().item()),
        }
        return loss, metrics
    
    def step_i2i(self, batch, celoss, device='cuda', eval=False):
        """
        One training/eval step for i2i-fd surrogate.

        Loss = lam_inr * ||pred_inr - target_inr||^2
            + lam_der * ||d_pred/d_x - d_target/d_x||^2  (FD only)
            + lam_cls * CrossEntropy(clf(embed(pred_inr)), label)
        """
        params, init_params, src, labels = batch
        labels = labels.to(device)
        src = src.to(device)  # (B, C, H, W) in [-1, 1]
        params = tuple([[i.to(device) for i in inr] for inr in params])
        # init_params = tuple([[i.to(device) for i in inr] for inr in init_params])
        batch_size = labels.shape[0]

        # Get target INR weights as flat vector
        target_wsfeat = WeightSpaceFeatures(*params).to(device)
        flat_target_inr = self.flatten_wsfeat(target_wsfeat)  # (B, total_params)

        # Surrogate prediction: Image -> INR weights
        if self.is_3d:
            pcd = farthest_point_sampling(src, 2048)
            flat_x = pcd.reshape(batch_size, -1)  # (B, 2048*3)
        else:
            flat_x = src.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        pred_inr_flat = self.surrogate(flat_x)  # (B, total_params)

        # INR reconstruction loss
        # loss_inr = (pred_inr_flat - flat_target_inr).pow(2).mean()
        loss_inr = torch.nn.functional.mse_loss(pred_inr_flat, flat_target_inr)
        pred_wsfeat = unflatten_to_wsfeat(pred_inr_flat, self.weight_shapes, self.bias_shapes)

        y_pred = self.victim(pred_wsfeat)
        loss_cls = celoss(y_pred, labels)

        if self.lam_recon > 0:
            if self.is_3d:
                # Generate UDF query points from PCD (same as inr2vec training)
                coords_list, labels_list = [], []
                for j in range(batch_size):
                    c, l = compute_udf_from_pcd(
                        src[j], self.num_queries_on_surface,
                        self.stds, self.num_points_per_std,
                        coords_range=(-1, 1), convert_to_bce_labels=True,
                    )
                    coords_list.append(c)
                    labels_list.append(l)
                query_coords = torch.stack(coords_list, dim=0).to(device)  # (B, N, 3)
                gt_labels = torch.stack(labels_list, dim=0).to(device)     # (B, N)
                
                pred_udf = self.evaluate_inr_on_coords(pred_wsfeat, query_coords)  # (B, N)
                loss_img_recon = torch.nn.functional.binary_cross_entropy_with_logits(pred_udf, gt_labels)
            else:
                pred_imgs = self.builder.build_img_from_wts_and_bs(pred_wsfeat, if_wsfeat=True, requires_grad=True)
                pred_imgs = pred_imgs.permute(0, 3, 1, 2) * 2.0 - 1.0
                loss_img_recon = torch.nn.functional.mse_loss(pred_imgs, src)
        else: 
            loss_img_recon = torch.tensor(0.0, device=device)
        loss = self.lam_inr * loss_inr + self.lam_cls * loss_cls + self.lam_recon * loss_img_recon

        loss_der = torch.tensor(0.0, device=device)
        
        if self.is_FD or self.test_only:
            # Weight-space perturbation for finite difference
            perturbed_wsfeat = target_wsfeat.map(lambda w: self.perturb_weight(w, alpha=self.alpha))
            perturbed_wsfeat = perturbed_wsfeat.to(device)
            flat_perturbed_inr = self.flatten_wsfeat(perturbed_wsfeat)

            if self.is_3d:
                # PCD space: no rendering needed for FD
                # delta_x is in PCD input space, use random direction
                delta_inr = ((flat_perturbed_inr - flat_target_inr) / self.alpha).detach()
                flat_delta_x = torch.randn_like(flat_x)
            else:
                with torch.no_grad():
                    # Render BOTH original and perturbed INR to correctly compute delta_x.
                    # Using src directly as the baseline introduces SIREN reconstruction error
                    # (|render(θ*) - src| / alpha), which gets amplified by 1/alpha.
                    original_imgs_recon = self.builder.build_img_from_wts_and_bs(
                        (target_wsfeat.weights, target_wsfeat.biases)
                    )
                    perturbed_imgs_recon = self.builder.build_img_from_wts_and_bs(
                        (perturbed_wsfeat.weights, perturbed_wsfeat.biases)
                    )
                    x_orig_recon = torch.permute(original_imgs_recon, (0, 3, 1, 2)).to(device) * 2.0 - 1.0
                    x_p = torch.permute(perturbed_imgs_recon, (0, 3, 1, 2)).to(device) * 2.0 - 1.0

                # Finite difference targets: pure J_render * Δθ, no reconstruction noise
                delta_x = ((x_p - x_orig_recon) / self.alpha).detach()  # (B, C, H, W)
                delta_inr = ((flat_perturbed_inr - flat_target_inr) / self.alpha).detach()  # (B, total_params)

                # Forward-mode AD to compute Jacobian-vector product
                flat_delta_x = delta_x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
            with fwAD.dual_level():
                dual_input = fwAD.make_dual(flat_x, flat_delta_x)
                _, d_hat = fwAD.unpack_dual(self.surrogate(dual_input))  # (B, total_params)

            loss_der = (d_hat - delta_inr).pow(2).mean()

            # OOD robustness: also match Jacobian at noise-augmented inputs so that
            # J_S(x+noise) ≈ J_fit, keeping gradient accurate at adversarial images.
            if not eval:
                if (self.n_aug > 0 and self.aug_eps > 0.0) or self.pgd_aug:
                    if self.pgd_aug:
                        src_aug = self.attack(src, labels)
                        self.victim.eval()
                    else:
                        noise = torch.empty(self.n_aug, *src.shape, device=src.device).uniform_(-self.aug_eps, self.aug_eps)
                        src_aug = (src.unsqueeze(0) + noise).clamp(-1.0, 1.0).flatten(0, 1)

                    n = 1 if self.pgd_aug else self.n_aug
                    if self.is_3d:
                        flat_x_aug = farthest_point_sampling(src_aug, 2048).reshape(-1, flat_x.shape[-1])
                    else:
                        flat_x_aug = src_aug.permute(0, 2, 3, 1).contiguous().view(-1, flat_x.shape[-1])
                    with fwAD.dual_level():
                        dual_aug = fwAD.make_dual(flat_x_aug, flat_delta_x.repeat(n, 1))
                        _, d_hat_aug = fwAD.unpack_dual(self.surrogate(dual_aug))
                    aug_loss = (d_hat_aug - delta_inr.repeat(n, 1)).pow(2).mean()
                    loss_der = (loss_der + aug_loss) * 0.5

            loss = loss + self.lam_der * loss_der

        metrics = {
            "loss": float(loss.item()),
            "inr": float(loss_inr.item()),
            "cls": float(loss_cls.item()),
            "der": float(loss_der.item()) if self.is_FD else 0.0,
            "recon": float(loss_img_recon.item()),
            "acc": float((y_pred.argmax(dim=1) == labels).float().mean().item()),
        }
        return loss, metrics
    
    def train_surrogate(self):

        opt = torch.optim.AdamW(self.surrogate.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        celoss = nn.CrossEntropyLoss()
        global_step = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        self.surrogate.train()
        timer = SimpleTimer(total_steps=len(self.trainloader)*(self.epochs - self.resume_epoch), start_step=global_step)
        # torch.cuda.reset_peak_memory_stats()

        for epoch in range(self.resume_epoch, self.epochs):
            epoch_loss = 0
            for b_idx, batch in enumerate(self.trainloader):
                opt.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=self.amp):
                    loss, metrics = self.step( batch=batch, celoss=celoss, )
                loss.backward()
                opt.step()

                if self.wandb_manager:
                    self.wandb_manager.log(metrics, step=global_step)

                epoch_loss += float(loss.item())
                global_step += 1

                if self.log_image_interval > 0 and (global_step % self.log_image_interval == 0) and not self.is_3d:
                    # Log some reconstructed images
                    self.surrogate.eval()
                    with torch.no_grad():
                        params, init_params, img, label = batch
                        
                        params = tuple([[i.to(self.device) for i in inr] for inr in params])
                        target_wsfeat = WeightSpaceFeatures(*params).to(self.device)

                        flat_x = img.permute(0, 2, 3, 1).contiguous().view(img.shape[0], -1).to(self.device)
                        flat_pred_inr = self.surrogate(flat_x)
                        pred_wsfeat = unflatten_to_wsfeat(flat_pred_inr, self.weight_shapes, self.bias_shapes)

                        self.builder.visualize_reconstruction(wts_and_bs=target_wsfeat, wts_and_bs_adv=pred_wsfeat, x=img, plot_dir=self.exp_plot_path, object="Predicted", title=f"train_{global_step}")
                    self.surrogate.train()

                if b_idx % 100 == 0 and b_idx > 0 and self.verbose:
                    eta = timer.get_eta(global_step)
                    print(f"Epoch {epoch} | Batch {b_idx}/{len(self.trainloader)} | Loss: {loss.item():.4f} | ETA: {eta}", flush=True)

                # surr_mem = torch.cuda.max_memory_allocated() / 1e9
                # print(f"Surrogate peak memory: {surr_mem:.1f} GB")
                # return 
            val_result = self.evaluate_surrogate(self.valloader)
            val_str = {k: f"{v:.4f}" for k, v in val_result.items()}
            print(
                f"Epoch {epoch} | Train Loss: {epoch_loss/len(self.trainloader):.4f} | "
                f"Val Result: {val_str}",
                flush=True
            )

            if self.wandb_manager:
                self.wandb_manager.log(val_result, step=global_step)

            if val_result['acc'] > best_val_accuracy:
                print(
                    f"Best val accuracy updated at epoch {epoch}: "
                    f"{best_val_accuracy:.4f} -> {val_result['acc']:.4f} | loss={val_result['loss']:.4f}",
                    flush=True
                )
                best_val_accuracy = val_result['acc']
                best_val_loss = val_result['loss']
                epochs_without_improvement = 0
                torch.save(self.surrogate.state_dict(), os.path.join(self.exp_ckpt_path, "best.pt"))
                
            elif epoch >= self.early_stop_from:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s) (patience={self.patience})", flush=True)
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best val accuracy: {best_val_accuracy:.4f}", flush=True)
                    break

        torch.save(self.surrogate.state_dict(), os.path.join(self.exp_ckpt_path, f"final.pt"))
        return best_val_accuracy, best_val_loss
    
    def evaluate_surrogate(self, dataloader):
        self.surrogate.eval()
        celoss = nn.CrossEntropyLoss()
        total_res = {}
        with torch.no_grad():
            for batch in dataloader:
                with autocast('cuda', enabled=False):
                    _, metrics = self.step( batch=batch, celoss=celoss, eval=True)

                for k, v in metrics.items():
                    if k not in total_res:
                        total_res[k] = 0.0
                    total_res[k] += float(v)
        self.surrogate.train()
        for k in total_res:
            total_res[k] /= max(len(dataloader), 1)
        return total_res    
    
    
def main(
         wandb_manager: Optional[WandbManager] = None,
         type='FD',
         **kwargs,
         ):
    
    trainer = SurrogateTrainer(**kwargs, type=type, wandb_manager=wandb_manager)

    exp_ckpt_path = kwargs['exp_ckpt_path']
    test_only = kwargs.get('test_only', False)
    has_final = os.path.exists(os.path.join(exp_ckpt_path, "final.pt"))
    has_best = os.path.exists(os.path.join(exp_ckpt_path, "best.pt"))
    resume = kwargs.get('surrogate_resume_training', False)

    if has_best and not resume:
        trainer.surrogate.load_state_dict(torch.load(os.path.join(exp_ckpt_path, "best.pt"), weights_only=False))
        print(f"Loaded surrogate from {os.path.join(exp_ckpt_path, 'best.pt')}")
        if test_only:
            test_result = trainer.evaluate_surrogate(trainer.testloader)
            print(test_result)
    else:
        print(f"Model will be saved to {exp_ckpt_path}")
        print(f"{'Resuming' if has_best else 'Starting'} training. Model will be saved to {exp_ckpt_path}")
        if has_best and resume:
            trainer.load_surrogate(path=os.path.join(exp_ckpt_path, "final.pt"))
        best_val_accuracy, best_val_loss = trainer.train_surrogate()
        test_result = trainer.evaluate_surrogate(trainer.testloader)

        print(
            f"Test accuracy: {test_result['acc']:.4f} | Test loss: {test_result['loss']:.4f} | "
            f"Best val accuracy: {best_val_accuracy:.4f} | Best val loss: {best_val_loss:.4f}"
        )
        if wandb_manager:
            wandb_manager.log_summary({
                "test/accuracy": test_result['acc'],
                "test/loss": test_result['loss'],
                "best_val/accuracy": best_val_accuracy,
                "best_val/loss": best_val_loss,
            })

    surrogate = SurrogateAttacker(is_i2i=trainer.is_i2i, 
                                  surrogate=trainer.surrogate, 
                                  victim=trainer.victim, 
                                  weight_shapes=trainer.weight_shapes, 
                                  bias_shapes=trainer.bias_shapes, 
                                  is_3d=trainer.is_3d
                                  )
    return surrogate

