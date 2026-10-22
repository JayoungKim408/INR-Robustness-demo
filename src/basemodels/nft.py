import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
import numpy as np
from datetime import timedelta
from torch import nn
from torch.utils import data
from datetime import datetime
from typing import Optional
from torch.utils.data import DataLoader, ConcatDataset



from basemodels.utils.nfn.train_utils import get_linear_warmup_with_cos_decay
from utils.wandb_manager import WandbManager
from utils.config_utils import config_to_dict


class SimpleTimer:
    def __init__(self, total_steps):
        self.total = total_steps
        self.start = time.time()

    def get_eta(self, current_step):
        elapsed = time.time() - self.start
        speed = current_step / elapsed if elapsed > 0 else 0
        eta = (self.total - current_step) / speed if speed > 0 else 0
        return str(timedelta(seconds=int(eta)))



class TransformerClassifier(nn.Module):
    def __init__(self, latent_dim=256, n_classes=10, num_blocks=12):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            512, 4, dim_feedforward=2048, dropout=0.1,
            activation="gelu", layer_norm_eps=1e-05,
            batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_blocks)
        self.proj = nn.Linear(latent_dim, 512)
        self.pos = nn.Embedding(64, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.proj(x)
        x = x + self.pos(torch.arange(0, x.shape[1]).to(x.device))
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.classifier(x)
        return x


def load_data(embedding_path):
    emb_data = torch.load(embedding_path, weights_only=False)
    embeddings = emb_data["embeddings"]
    labels = emb_data["labels"]
    split_points = emb_data["split_points"]
    return embeddings, labels, split_points


def make_embeddings(inr2array_cfg):

    if not os.path.exists(os.path.join(inr2array_cfg['embedding_dir'], "embeddings.pt")):
        from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
        from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset
        from basemodels.inr2array import AutoEncoder, PerceiverPooling
        from functools import partial

        # load data
        print('\n\n\n', inr2array_cfg['dset']['siren_path'], inr2array_cfg['dset']['data_path'])
        traindset = get_SirenAdnOriginalDataset(**inr2array_cfg['dset'], split='train')
        valdset = get_SirenAdnOriginalDataset(**inr2array_cfg['dset'], split='val')
        testdset = get_SirenAdnOriginalDataset(**inr2array_cfg['dset'], split='test')

        extra_aug = inr2array_cfg.get('extra_aug', True)
        n_extra_aug = inr2array_cfg.get('n_extra_aug', 10)
        aug_dsets = []
        if extra_aug:
            for i in range(1, n_extra_aug):
                aug_dsets.append(get_SirenAdnOriginalDataset(
                    siren_path=inr2array_cfg['dset']['siren_path'],
                    siren_prefix=f"{inr2array_cfg['dset']['siren_prefix']}_{i}",
                    data_path=inr2array_cfg['dset']['data_path'],
                    split_points=inr2array_cfg['dset']['split_points'],
                    split="train",
                ))

        dset = ConcatDataset([traindset] + aug_dsets + [valdset] + [testdset])
        loader = DataLoader(dset, batch_size=1024, shuffle=False, num_workers=8, drop_last=False)
        print(f"Dataset sizes: total={len(dset)}, batch_size={inr2array_cfg['bs']}.")

        # load weight features
        spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(loader))[0]).to("cpu"), set_all_dims=True)

        # Create model directly without hydra.utils.instantiate
        model_cfg = dict(inr2array_cfg.get('model', {}))
        # Extract pool_cls from model_cfg (old Hydra format has it nested in model)
        pool_cfg_raw = model_cfg.pop('pool_cls', None) or inr2array_cfg.get('pool_cls', None)
        # Convert to plain dict and remove Hydra-specific keys
        if pool_cfg_raw is not None:
            pool_cfg = dict(pool_cfg_raw) if hasattr(pool_cfg_raw, 'items') else {}
            pool_cfg.pop('_target_', None)
            pool_cfg.pop('_partial_', None)
        else:
            pool_cfg = {}
        model_cfg.pop('_target_', None)
        # Fix config inconsistency: spatial=True requires reduce=False
        if model_cfg.get('spatial', False) and pool_cfg.get('reduce', False):
            pool_cfg['reduce'] = False
        pool_cls = partial(PerceiverPooling, **pool_cfg) if pool_cfg else None
        nfnet = AutoEncoder(model_cfg, spec, dset_data_type=inr2array_cfg['dataset_name'], pool_cls=pool_cls).to("cuda")
        nfnet.load_state_dict(torch.load(os.path.join(inr2array_cfg['inr2array_model_dir'], "best.pt"), weights_only=False))

        nfnet.eval()
    
        for param in nfnet.parameters():
            param.requires_grad = False

        print("Computing embeddings...")
        embeddings, labels = [], []
        # params, init_params, img, siren_label
        for wts_and_bs, init_wts_and_bs, img, label in loader:
            params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
            with torch.no_grad():
                embeddings.append(nfnet.encode(params).cpu())
                labels.append(label)
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)

        if inr2array_cfg['extra_aug']:
            split_points = [inr2array_cfg['dset']['split_points'][0] * inr2array_cfg['n_extra_aug'], inr2array_cfg['dset']['split_points'][0]*inr2array_cfg['n_extra_aug'] + 5000]
        else: 
            if inr2array_cfg['dset']['split_points'] is None:
                split_points = [len(traindset), len(traindset) + len(valdset)]
            else:
                split_points = inr2array_cfg['dset']['split_points']
            
        torch.save({
            "embeddings": embeddings,
            "labels": labels,
            "split_points": split_points
        }, os.path.join(inr2array_cfg['embedding_dir'], "embeddings.pt"))

        print(f"Split points: {split_points}")
        print("Embeddings saved to", os.path.join(inr2array_cfg['embedding_dir'], "embeddings.pt"))
    else:
        print("Embeddings already exists, not doing anything.")
        out_data = torch.load(os.path.join(inr2array_cfg['embedding_dir'], "embeddings.pt"))
        embeddings = out_data["embeddings"]
        labels = out_data["labels"]
        split_points = out_data["split_points"]
    return embeddings, labels, split_points
    
    
def make_dataset(embeddings, labels, split_points, bs):
    # construct tensordataset from embeddings and labels
    dataset = data.TensorDataset(embeddings, labels)
    # split into train/val/test sets.
    val_point, test_point = split_points
    trainset, valset = data.Subset(dataset, range(val_point)), data.Subset(dataset, range(val_point, test_point))
    testset = data.Subset(dataset, range(test_point, len(dataset)))
    trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True, drop_last=True)
    valloader = data.DataLoader(valset, batch_size=bs, shuffle=False, drop_last=True)
    testloader = data.DataLoader(testset, batch_size=bs, shuffle=False, drop_last=True)
    print(f"Dataset sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}, batch_size={bs}.")
    return trainloader, valloader, testloader


@torch.no_grad()
def evaluate(loader, model):
    state = model.training
    model.eval()
    acc = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
    model.train(state)
    return acc / len(loader)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_and_evaluate(
    clf, opt, sched,
    trainloader,
    valloader,
    testloader,
    n_epochs=10,
    out_dir="",
    mixup_alpha=0,
):
    loss_fn = nn.CrossEntropyLoss()
    best_ckpt_path = os.path.join(out_dir, "best.pt")
    if os.path.exists(best_ckpt_path):
        print("Checkpoint exists. Evaluating...")
    else:
        global_step = 0
        train_acc, total_train_loss = 0, 0
        total_val_acc, best_val_acc = 0, 0
        timer = SimpleTimer(n_epochs)
        for epoch_idx in range(n_epochs):
            for x, y in trainloader:
                opt.zero_grad()
                x, y = x.cuda(), y.cuda()
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                logits = clf(x)
                loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
                loss.backward()
                opt.step()
                if sched is not None: sched.step()
                global_step += 1
                if global_step % 1000 == 0:
                    train_acc_a = (logits.detach().argmax(dim=-1).cpu().numpy() == y_a.detach().cpu().numpy()).mean().item()
                    train_acc_b = (logits.detach().argmax(dim=-1).cpu().numpy() == y_b.detach().cpu().numpy()).mean().item()
                    train_acc = lam * train_acc_a + (1 - lam) * train_acc_b
                    if wandb.run:
                        wandb.log({"train/loss": loss.detach().item(), "train/acc": train_acc}, step=global_step)
                    # calculate val accuracy
                    val_acc = evaluate(valloader, clf)
                    total_val_acc += val_acc
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save({"clf":clf.state_dict(), "global_step":global_step}, best_ckpt_path)
                    print(f"Epoch {epoch_idx} | Step {global_step} | ETA: {timer.get_eta(epoch_idx)} | train loss: {loss.item():.3f} | train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | best val acc: {best_val_acc:.3f}")
                total_train_loss += loss.item()
        print(f"Epoch {epoch_idx} | ETA: {timer.get_eta(epoch_idx)} | train loss: {total_train_loss/len(trainloader):.3f} | val acc: {total_val_acc/len(valloader):.3f} | best val acc: {best_val_acc:.3f}")
        if wandb.run:
            wandb.log({"val/acc": val_acc, "val/best_acc": best_val_acc}, step=global_step)
        # outer_pbar.close()
    final_train_acc = evaluate(trainloader, clf)
    # load best.pt
    loaded_dict = torch.load(best_ckpt_path, weights_only=False)
    if "clf" in loaded_dict:
        clf.load_state_dict(loaded_dict['clf'])
        global_step = loaded_dict['global_step']
    else:
        clf.load_state_dict(loaded_dict)
        global_step = 0
    test_acc = evaluate(testloader, clf)
    print(f"Test accuracy: {test_acc:.3f}. Final train_acc: {final_train_acc:.3f} | {best_ckpt_path}")
    if wandb.run:
        wandb.log({"final_train/acc": final_train_acc, "test/acc": test_acc}, step=global_step)


# def main(cfg):
def main(cfg: dict, wandb_manager: Optional[WandbManager] = None):

    wandb_id = None
    # Initialize wandb run
    if wandb_manager.enabled and wandb_manager.current_run is None:
        run_name = wandb_manager.generate_run_name(
            dataset=cfg.dataset_name,
            model="nft",
            training_type="inr2array",
        )
        wandb_manager.init_run(
            name=run_name,
            group=f"inr2array_{cfg.dataset_name}",
            tags=wandb_manager.generate_tags(
                dataset=cfg.dataset_name,
                model="nft",
                training_type="inr2array",
            ),
            config=config_to_dict(cfg),
            run_id=wandb_id,
            resume="must" if wandb_id else None,
            job_type="training",
        )
        
    embeddings, labels, split_points = make_embeddings(cfg['inr2array'])
    trainloader, valloader, testloader = make_dataset(embeddings, labels, split_points, cfg['inr2array']['bs'])
    if "pool_cls" in cfg['inr2array'] and "latent_dim" in cfg['inr2array']['pool_cls']:
        latent_dim = cfg['inr2array']['pool_cls']['latent_dim']
    clf = TransformerClassifier(latent_dim=latent_dim).cuda()
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    total_steps = len(trainloader) * cfg['n_epochs']
    sched = get_linear_warmup_with_cos_decay(opt, total_steps=total_steps, warmup_steps=10_000)
    # train
    start_time = datetime.now()
    train_and_evaluate(clf, opt, sched, trainloader, valloader, testloader, n_epochs=cfg['n_epochs'], out_dir=cfg['latent_clf_dir'], mixup_alpha=cfg['alpha'])
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed}")