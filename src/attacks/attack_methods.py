"""Attack method implementations for image, INR, and embedding spaces."""

import os
from typing import Any, Dict, Optional
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
from tqdm import tqdm

from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from basemodels.inr2array import AutoEncoder

from attacks.utils import BatchedINRBuilder, wsfeat_to_dws, DEFAULT_EPS, DEFAULT_EPS_STD, assert_within_eps
# from attacks.surrogate.train_surrogate import I2ISurrogateAttack, surrogate_attack
from attacks.torchattacks.attacks.pgd import PGD
from attacks.torchattacks.attacks.pgd3d import PGD3D
from attacks.torchattacks.attacks.pgdl1 import PGDL1
from attacks.torchattacks.attacks.pgdl2 import PGDL2
from attacks.torchattacks.attacks.deepfool import DeepFool
from attacks.torchattacks.attacks.pgdwsfeat import PGDWSFeat
from attacks.torchattacks.attacks.pgdembedding import PGDEmbedding, compute_embedding_stats
from attacks.scalegmn_utils import wsfeat_to_pyg_batch, get_scalegmn_batch_params
from basemodels.model_loading import load_victim_model, load_testloader


def get_target_eps(norm, eps, dataset_name):
    sidelen = 32 if dataset_name == "cifar" else 28
    c_out = 3 if dataset_name == "cifar" else 1
    
    if norm == 'l2':
        target_eps = math.sqrt(sidelen * sidelen * c_out) * eps
    elif norm == 'l1':
        target_eps = sidelen * sidelen * c_out * eps
    elif norm == 'linf':
        target_eps = eps

    return target_eps

def get_attacker(surrogate, dataset_name, kwargs):
    ATTACK_MAP = {"pgd": {
                            "linf": PGD,
                            "l2": PGDL2,
                            "l1": PGDL1,
                         },
                "inr": PGDWSFeat,
                "embedding": PGDEmbedding,
                "3d": PGD3D,
                "deepfool": DeepFool}
    # Set PGD
    eps = kwargs.get("eps", DEFAULT_EPS[dataset_name])
    steps = kwargs.get("steps", 20)
    norm = kwargs.get("norm", "linf")

    eps_scaled = eps * 2  # for [-1, 1] range
    target_eps = get_target_eps(norm, eps_scaled, dataset_name)
    gamma = 2 * target_eps / steps


    attack_args = {
        "model": surrogate,
        "eps": target_eps,
        "steps": steps,
        "alpha": gamma,
    }
    if kwargs['attack_space'] == "inr":
        attack_args['renderer'] = kwargs.get("renderer")
        attack_args['eps_img'] = target_eps
    else:
        attack_args["img_scale"] = [-1, 1]
    
    attack = ATTACK_MAP[kwargs['attack_type']](**attack_args) if kwargs['attack_type'] != "pgd" else ATTACK_MAP["pgd"][norm](**attack_args)
    
    print(f"[{kwargs['attack_type'].upper()} Attack] {kwargs['attack_type'].upper()} attack initialized with eps={eps}, alpha={gamma}, steps={steps}, norm={norm}")
    return attack


def run_evaluate_classifiers(
        device='cuda', 
        **kwargs):
    victim = kwargs["victim_wrapped"]
    victim.eval()
    attack_emb = True if kwargs['attack_space'] == "embedding" else False

    _, _, testloader = kwargs['dataloader']

    clean_correct, total = 0, 0

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
            embs = None

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)

            if attack_emb:
                inp = embs.to(device, non_blocking=True)
            else:
                inp = WeightSpaceFeatures(*inrs).to(device)

            clean_pred = victim(inp).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
            total += labels.numel()

    final_clean_acc = clean_correct / max(total, 1)

    return {
        "cls_acc": final_clean_acc,
    }

    
def run_pgd(
        dataset_name,
        device='cuda', 
        save_intermediate_results=False,
        **kwargs):
    # Load victim
    victim = kwargs["victim_wrapped"]
    victim.eval()

    # Load testset loader
    _, _, testloader = kwargs['dataloader']

    # Load surrogate
    surrogate = kwargs['surrogate'] 
    surrogate.eval()

    n_restarts = kwargs.get("n_restarts", 5)
    attack = get_attacker(surrogate, dataset_name, kwargs)
    is_3d = dataset_name == "shapenet"

    # Set builder
    builder = kwargs['builder']
    def build_image(init_params, imgs):
        wts_and_bs = builder.build_wts_and_bs_from_img(imgs, WeightSpaceFeatures(*init_params))
        return WeightSpaceFeatures(*wts_and_bs).to(device), wts_and_bs
    
    clean_correct, adv_correct, total = 0, 0, 0

    if save_intermediate_results:
        total_adv_samples = []
        
    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
            embs = None

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        # Multi-restart PGD: run n_restarts times, keep per-sample worst case
        best_adv_srcs = srcs.clone()
        best_loss = torch.full((srcs.shape[0],), -float('inf'), device=device)
        expand_shape = [-1] + [1] * (srcs.dim() - 1)

        for _ in range(n_restarts):
            with torch.enable_grad():
                adv_srcs_r = attack(srcs, labels)
            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
                    surrogate(adv_srcs_r), labels
                )
            update_mask = per_sample_loss > best_loss
            best_adv_srcs = torch.where(update_mask.view(*expand_shape), adv_srcs_r, best_adv_srcs)
            best_loss = torch.where(update_mask, per_sample_loss, best_loss)

        adv_srcs = best_adv_srcs
        with torch.no_grad():
            if is_3d:
                inp = WeightSpaceFeatures(*inrs).to(device)
                adv_inp, _ = builder.build_wts_and_bs_from_pcd(adv_srcs, WeightSpaceFeatures(*init_params))
                adv_inp = WeightSpaceFeatures(*adv_inp).to(device)

            else:
                inp, wts_and_bs = build_image(init_params, srcs)
                adv_inp, wts_and_bs_adv = build_image(init_params, adv_srcs)

            clean_pred = victim(inp).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = victim(adv_inp).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

        if save_intermediate_results:
            total_adv_samples.append({"clean_img": imgs.cpu(), "adv_img": adv_srcs.cpu(), "clean_wsfeat": wts_and_bs, "adv_wsfeat": wts_and_bs_adv, "label": labels.cpu()})

        # Visualization every 10 batches
        if (i % 10 == 1):
            builder.visualize_reconstruction(wts_and_bs=inp, wts_and_bs_adv=adv_inp, x=imgs, x_adv=adv_srcs, eps=attack.eps, steps=attack.steps, norm=kwargs.get('norm', 'linf'), plot_dir=kwargs['exp_plot_path'], object="Perturbed", title=f"attack_{i}")
        print(
            f"BATCH {i} | Clean acc: {clean_correct / max(total, 1):.4f} | "
            f"Adv acc: {adv_correct / max(total, 1):.4f} ({adv_correct})"
        )
    if save_intermediate_results:
        torch.save(total_adv_samples, os.path.join(kwargs['path_config'].base_path, "attack_intermediate_results.pt"))

    final_clean_acc = clean_correct / max(total, 1)
    final_adv_acc = adv_correct / max(total, 1)

    return {
        "clean_acc": final_clean_acc,
        "adv_acc": final_adv_acc,
        "total": total
    }


def run_pgd_against_surrogate(
        dataset_name,
        device='cuda', 
        save_intermediate_results=False,
        **kwargs):
    
    if save_intermediate_results:
        from attacks.hypernet.hypernet_attack import convert_to_state_dicts
        from basemodels.utils.nfn.nfn.common import state_dict_to_tensors
        from torch.utils.data import default_collate

        def get_hypernet_prediction(img):
            inr, _ = surrogate.surrogate.get_hypo_net_weights({'img_sparse': img})
            hypo_params_state_dicts = convert_to_state_dicts(inr)
            wts_and_bs = [state_dict_to_tensors(sd) for sd in hypo_params_state_dicts]
            wts_and_bs = default_collate(wts_and_bs)
            return wts_and_bs

    # Load victim
    victim = kwargs["victim_wrapped"]
    victim.eval()

    # Load testset loader
    _, _, testloader = kwargs['dataloader']

    # Load surrogate
    surrogate = kwargs['surrogate'] 
    surrogate.eval()

    n_restarts = kwargs.get("n_restarts", 1)
    attack = get_attacker(surrogate, dataset_name, kwargs)
    clean_correct, adv_correct, total = 0, 0, 0

    if save_intermediate_results:
        res = []
    # torch.cuda.reset_peak_memory_stats()

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        # Multi-restart PGD: run n_restarts times, keep per-sample worst case
        best_adv_srcs = srcs.clone()
        best_loss = torch.full((srcs.shape[0],), -float('inf'), device=device)
        expand_shape = [-1] + [1] * (srcs.dim() - 1)

        for _ in range(n_restarts):
            with torch.enable_grad():
                adv_srcs_r = attack(srcs, labels)
            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
                    surrogate(adv_srcs_r), labels
                )
            update_mask = per_sample_loss > best_loss
            best_adv_srcs = torch.where(update_mask.view(*expand_shape), adv_srcs_r, best_adv_srcs)
            best_loss = torch.where(update_mask, per_sample_loss, best_loss)

        adv_srcs = best_adv_srcs

        with torch.no_grad():
            clean_pred = surrogate(srcs).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = surrogate(adv_srcs).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

        print(
            f"BATCH {i} | Clean acc: {clean_correct / max(total, 1):.4f} | "
            f"Adv acc against surrogate: {adv_correct / max(total, 1):.4f} ({adv_correct})"
        )
        if save_intermediate_results:
            res.append({"clean_img": srcs.cpu(), "adv_img": adv_srcs.cpu(), "label": labels.cpu()})
        # print(f"Surrogate: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        # return 

    final_clean_acc = clean_correct / max(total, 1)
    final_adv_acc = adv_correct / max(total, 1)
    print(f"Final Clean Acc: {final_clean_acc:.4f}, Final Adv Acc against surrogate: {final_adv_acc:.4f}")
    return {
        "surrogate_clean_acc": final_clean_acc,
        "surrogate_adv_acc": final_adv_acc,
    }


 
def run_inr_autoattack(
        dataset_name,
        device='cuda',
        **kwargs):
    """
    Run AutoAttack in INR weight space.
    Evaluates APGD-CE, APGD-DLR, and Square Attack sequentially.
 
    Args:
        dataset_name: 'mnist', 'fashion', 'cifar', etc.
        device: torch device
        **kwargs: must contain:
            - victim_wrapped: classifier
            - dataloader: (train, val, test) loaders
            - renderer: INR renderer from get_batch_inr()
 
    Returns:
        dict with clean_acc, per-attack robust acc, final_robust_acc
    """
    from attacks.torchattacks.autoattack.inr_autoattack import AutoAttackINR

    from attacks.utils import DEFAULT_EPS
 
    victim = kwargs["victim_wrapped"]
    victim.eval()
    _, _, testloader = kwargs['dataloader']
    renderer = kwargs.get("renderer")
 
    eps = kwargs.get("eps", DEFAULT_EPS[dataset_name])
    eps_scaled = eps * 2  # scale for [-1, 1] range
    n_iter = kwargs.get("n_iter", 100)
    n_restarts = kwargs.get("n_restarts", 5)
    n_queries = kwargs.get("n_queries", 100)
    attacks_to_run = kwargs.get("attacks_to_run", ['apgd-ce', 'apgd-dlr', 'square'])
 
    aa = AutoAttackINR(
        model=victim,
        renderer=renderer,
        eps=eps_scaled,
        n_iter=n_iter,
        n_restarts=n_restarts,
        n_queries=n_queries,
        attacks_to_run=attacks_to_run,
        verbose=True,
        device=device,
    )
 
    # Accumulate results across batches
    all_results = {}
    total_clean_correct = 0
    total_robust = {attack: 0 for attack in attacks_to_run}
    total_final_robust = 0
    total_samples = 0
 
    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
 
        labels = labels.to(device, non_blocking=True)
        wsfeat = WeightSpaceFeatures(*inrs).to(device)
 
        # Run AutoAttack on this batch
        batch_results = aa.run_evaluation(wsfeat, labels)

        B = labels.shape[0]
        total_samples += B
        total_clean_correct += int(batch_results['clean_acc'] * B)
        total_final_robust += int(batch_results['final_robust_acc'] * B)
 
        for attack in attacks_to_run:
            if attack in batch_results:
                total_robust[attack] += int(batch_results[attack] * B)
 
        print(f"\n--- Batch {i} cumulative ---")
        print(f"  Clean acc: {total_clean_correct/total_samples:.4f}")
        for attack in attacks_to_run:
            print(f"  After {attack}: {total_robust[attack]/total_samples:.4f}")
        print(f"  Final robust acc: {total_final_robust/total_samples:.4f}")
 
    final_results = {
        'clean_acc': total_clean_correct / max(total_samples, 1),
        'final_robust_acc': total_final_robust / max(total_samples, 1),
        'total': total_samples,
    }
    for attack in attacks_to_run:
        final_results[f'after_{attack}'] = total_robust[attack] / max(total_samples, 1)
 
    print(f"\n{'='*60}")
    print(f"AutoAttack-INR Final Results ({total_samples} samples)")
    print(f"  Clean accuracy:       {final_results['clean_acc']:.4f}")
    for attack in attacks_to_run:
        print(f"  After {attack:12s}:  {final_results[f'after_{attack}']:.4f}")
    print(f"  Final robust accuracy: {final_results['final_robust_acc']:.4f}")
    print(f"{'='*60}")
 
    return final_results
 

def run_square_attack(
        dataset_name,
        device='cuda',
        **kwargs):
    from attacks.torchattacks.autoattack.square_attack import SquareAttack
    
    victim = kwargs["victim_wrapped"]
    victim.eval()
    _, _, testloader = kwargs['dataloader']
    indices = torch.randperm(len(testloader.dataset))[:1000].tolist()
    subset = Subset(testloader.dataset, indices)
    testloader = DataLoader(subset, batch_size=testloader.batch_size, shuffle=False)

    builder = kwargs['builder']
    
    eps = kwargs.get("eps", DEFAULT_EPS[dataset_name])
    eps_scaled = eps * 2
    norm = kwargs.get("norm", "Linf")
    n_queries = kwargs.get("n_queries", 100)
    
    class INRPipelineWrapper(nn.Module):
        def __init__(self, builder, victim):
            super().__init__()
            self.builder = builder
            self.victim = victim
            self.current_init_params = None
            
        def forward(self, imgs):
            wts_and_bs = self.builder.build_wts_and_bs_from_img(
                imgs, self.current_init_params
            )
            inp = WeightSpaceFeatures(*wts_and_bs).to(imgs.device)
            return self.victim(inp)
    
    pipeline = INRPipelineWrapper(builder, victim)
    
    clean_correct, adv_correct, total = 0, 0, 0
    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
        
        labels = labels.to(device, non_blocking=True)
        srcs = imgs.to(device, non_blocking=True)
        full_init_params = WeightSpaceFeatures(*init_params).to(device)
        
        # Step 1: Find correctly classified samples
        pipeline.current_init_params = full_init_params
        with torch.no_grad():
            clean_logits = pipeline(srcs)
            clean_pred = clean_logits.argmax(dim=1)
            correct_mask = (clean_pred == labels)
        
        clean_correct += correct_mask.sum().item()
        total += labels.numel()
        
        if correct_mask.sum() == 0:
            print(f"BATCH {i} | No correct samples, skipping attack")
            continue
        
        # Step 2: Only attack correctly classified samples
        square = SquareAttack(
            predict=pipeline,
            eps=eps_scaled,
            norm='Linf',
            n_queries=n_queries,
            verbose=True,
            device=device
        )
        adv_srcs = square.perturb(srcs, labels)

        # Step 3: Evaluate adversarial samples
        with torch.no_grad():
            adv_pred = pipeline(adv_srcs).argmax(dim=1)
            # Only count samples that were correctly classified AND survived attack
            adv_correct += (adv_pred[correct_mask] == labels[correct_mask]).sum().item()

        print(
            f"BATCH {i} | Clean acc: {clean_correct/max(total,1):.4f} | "
            f"Adv acc (Square): {adv_correct/max(total,1):.4f}"
        )
    
    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
        "total": total,
    }

def run_inr_square_attack(
        dataset_name,
        device='cuda',
        **kwargs):
    """
    Square Attack in INR weight space.
    No surrogate needed, no init needed.
    Attacks fitted INR weights directly.
    """
    from attacks.torchattacks.autoattack.inr_square_attack import SquareAttackINR
    
    victim = kwargs["victim_wrapped"]
    victim.eval()
    _, _, testloader = kwargs['dataloader']
    
    eps = kwargs.get("eps", DEFAULT_EPS[dataset_name])
    eps_scaled = eps * 2  # for [-1, 1] range
    n_queries = kwargs.get("n_queries", 1000)
    renderer = kwargs.get("renderer")  # batched INR renderer
    
    square = SquareAttackINR(
        model=victim,
        renderer=renderer,
        eps=eps_scaled,
        n_queries=n_queries,
        verbose=True,
        device=device
    )
    
    clean_correct, adv_correct, total = 0, 0, 0
    
    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
        
        labels = labels.to(device, non_blocking=True)
        wsfeat = WeightSpaceFeatures(*inrs).to(device)
        
        # Clean accuracy
        with torch.no_grad():
            clean_pred = victim(wsfeat).argmax(dim=1)
            correct_mask = (clean_pred == labels)
            clean_correct += correct_mask.sum().item()
        
        # Square Attack on INR weights
        adv_wsfeat = square.perturb(wsfeat, labels)
        
        with torch.no_grad():
            adv_pred = victim(adv_wsfeat).argmax(dim=1)
            adv_correct += (adv_pred[correct_mask] == labels[correct_mask]).sum().item()
        
        total += labels.numel()
        print(
            f"BATCH {i} | Clean acc: {clean_correct/max(total,1):.4f} | "
            f"Adv acc (Square INR): {adv_correct/max(total,1):.4f}"
        )
    
    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
        "total": total,
    }


def run_deepfool_against_surrogate(
        dataset_name,
        device='cuda',
        save_intermediate_results=False,
        **kwargs):

    from attacks.torchattacks.attacks.deepfool import DeepFool
    from tqdm import tqdm
    import torch
    import torch.nn as nn

    # Load victim
    victim = kwargs["victim_wrapped"]
    victim.eval()

    # Load testset loader
    _, _, testloader = kwargs['dataloader']

    # Load surrogate
    surrogate = kwargs['surrogate']
    surrogate.eval()

    # DeepFool params
    steps = kwargs.get("deepfool_steps", 50)
    overshoot = kwargs.get("deepfool_overshoot", 0.02)
    img_scale = kwargs.get("img_scale", [-1, 1])

    attack = DeepFool(model=surrogate, steps=steps, overshoot=overshoot, img_scale=img_scale)

    clean_correct, adv_correct, total = 0, 0, 0

    if save_intermediate_results:
        res = []

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch
        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        # DeepFool is deterministic — no multi-restart needed
        with torch.enable_grad():
            adv_srcs = attack(srcs, labels)

        with torch.no_grad():
            clean_pred = surrogate(srcs).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = surrogate(adv_srcs).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

        # Compute L2 perturbation stats
        l2_norms = torch.norm(
            (adv_srcs - srcs).reshape(srcs.shape[0], -1), p=2, dim=1
        )
        mean_l2 = l2_norms.mean().item()
        max_l2 = l2_norms.max().item()

        print(
            f"BATCH {i} | Clean acc: {clean_correct / max(total, 1):.4f} | "
            f"Adv acc against surrogate: {adv_correct / max(total, 1):.4f} ({adv_correct}) | "
            f"L2 mean: {mean_l2:.4f} max: {max_l2:.4f}"
        )
        if save_intermediate_results:
            res.append({"clean_img": srcs.cpu(), "adv_img": adv_srcs.cpu(), "label": labels.cpu()})

    final_clean_acc = clean_correct / max(total, 1)
    final_adv_acc = adv_correct / max(total, 1)
    print(f"Final Clean Acc: {final_clean_acc:.4f}, Final Adv Acc against surrogate: {final_adv_acc:.4f}")
    return {
        "surrogate_clean_acc": final_clean_acc,
        "surrogate_adv_acc": final_adv_acc,
    }


def run_image_gaussian_attack(ctx, basemodel_ckpt_path: Optional[str], **kwargs: Any) -> Dict[str, Any]:

    # if ctx.attack_type not in ("gaussian", "transfer"):
    #     raise ValueError(f"run_image_perturbation_attack only supports gaussian/transfer, got: {ctx.attack_type}")
    embedding_path = ctx.path_config.get_embedding_path(ctx.victim_model, ctx.dataset_name, ctx.inr_config, latent_dim=kwargs.get("latent_dim", 256))
    print(f"Using embedding path: {embedding_path}")

    # _, testloader = load_data(ctx.dataset_name, data_path=ctx.path_config.data_path)
    testloader = load_testloader(siren_path=ctx.inr_config.get_inr_path(ctx.path_config.base_path, ctx.dataset_name),
                                 siren_prefix=ctx.inr_config.prefix, 
                                 data_path=ctx.path_config.data_path, 
                                 dataset_name=ctx.dataset_name, 
                                 embedding_path=embedding_path)

    victim = load_victim_model(
        dataset_name=ctx.dataset_name, 
        device=ctx.device, 
        testloader=testloader, 
        return_wrapped=True, 
        ckpt_path=basemodel_ckpt_path, 
        **kwargs,
    )

    sidelen = 32 if ctx.dataset_name == "cifar" else 28
    c_out = 3 if ctx.dataset_name == "cifar" else 1

    builder = BatchedINRBuilder(
        sidelen=sidelen,
        hidden_features=ctx.inr_config.width,
        hidden_layers=ctx.inr_config.depth,
        first_omega_0=ctx.inr_config.scale,
        hidden_omega_0=ctx.inr_config.scale,
        c_out=c_out,
    )

    # attack 
    def _l2_projection(images, adv_images, eps):
        delta = adv_images - images
        delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
        factor = eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

        adv_images = torch.clamp(images + delta, min=-1.0, max=1.0).detach()
        return adv_images
    
    def _linf_projection(images, adv_images, eps):
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
        return adv_images

    eps = kwargs.get("eps", DEFAULT_EPS[ctx.dataset_name] * 2) # because input image is scaled to [-1, 1]
    eps_l2 = get_target_eps(norm='l2', eps=eps, dataset_name=ctx.dataset_name)

    clean_correct, adv_correct, total = 0, 0, 0

    for i, batch in enumerate(testloader):
        if len(batch) == 5:
            inrs, init_params, img, embs, labels = batch
        else:
            inrs, init_params, img, labels = batch
            embs = None
        img = img.to(ctx.device, non_blocking=True)
        label = labels.to(ctx.device, non_blocking=True)

        #TODO: handle inr and embedding space attacks
        img_adv = img + torch.randn_like(img) * eps
        # img_adv = _l2_projection(img, img_adv, eps_l2)
        # report, _, _ = assert_within_eps(img, img_adv, eps_l2, norm='l2') # l2 only
        img_adv = _linf_projection(img, img_adv, eps)
        report, _, _ = assert_within_eps(img, img_adv, eps, norm='linf') # l2 only

        print(report)

        # evaluate clean vs adversarial on the victim model
        wts_and_bs = builder.build_wts_and_bs_from_img(img, WeightSpaceFeatures(*init_params))
        wsfeat = WeightSpaceFeatures(*wts_and_bs).to(ctx.device)
        clean_pred = victim(wsfeat).argmax(dim=1)
        clean_correct += (clean_pred == label).sum().item()

        # build inr from perturbed image
        wts_and_bs_adv = builder.build_wts_and_bs_from_img(img_adv, WeightSpaceFeatures(*init_params))
        wsfeat_adv = WeightSpaceFeatures(*wts_and_bs_adv).to(ctx.device)
        adv_pred = victim(wsfeat_adv).argmax(dim=1)
        adv_correct += (adv_pred == label).sum().item()

        total += label.numel()
        print(f"Batch {i}: Clean Acc {(clean_correct/total)*100:.2f}%, Adv Acc {(adv_correct/total)*100:.2f}%")

        if i % 10 == 0:
            builder.visualize_reconstruction(wts_and_bs=wsfeat, 
                                            wts_and_bs_adv=wsfeat_adv, 
                                            x=img, 
                                            x_adv=img_adv, 
                                            eps=eps_l2,
                                            steps=1,
                                            norm="l2",
                                            plot_dir=ctx.exp_plot_path, 
                                            title=f"attack_{i}")

    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
    }


def run_image_transfer_attack(ctx, **kwargs: Any) -> Dict[str, Any]:
    from attacks.transfer.train_transfer_model.resnet import ResNet18

    embedding_path = ctx.path_config.get_embedding_path(ctx.victim_model, ctx.dataset_name, ctx.inr_config, latent_dim=kwargs.get("latent_dim", 256))
    print(f"Using embedding path: {embedding_path}")

    testloader = load_testloader(siren_path=ctx.inr_config.get_inr_path(ctx.path_config.base_path, ctx.dataset_name),
                                 siren_prefix=ctx.inr_config.prefix, 
                                 data_path=ctx.path_config.data_path, 
                                 dataset_name=ctx.dataset_name, 
                                 embedding_path=embedding_path)

    victim = kwargs['victim_wrapped']
    builder = kwargs["builder"]
    
    transfernet = ResNet18(in_c=3 if ctx.dataset_name == "cifar" else 1).to(ctx.device)
    tn_ckpt = torch.load(os.path.join(ctx.path_config.base_path, "assets", "attacks", "transfer", f"{ctx.dataset_name}.pth"), map_location="cpu", weights_only=False)["net"]
    tn_ckpt = {k.replace("module.", ""): v for k, v in tn_ckpt.items()}
    transfernet.load_state_dict(tn_ckpt)
    transfernet.eval()
    
    attack = get_attacker(transfernet, ctx.dataset_name, kwargs)
    clean_correct, adv_correct, total = 0, 0, 0

    for i, batch in enumerate(testloader):
        if len(batch) == 5:
            inrs, init_params, img, embs, label = batch
        else:
            inrs, init_params, img, label = batch
            embs = None
        img = img.to(ctx.device, non_blocking=True)
        label = label.to(ctx.device, non_blocking=True)

        #TODO: handle inr and embedding space attacks
        with torch.enable_grad():
            img_adv = attack(img, label)
            # report, stats, arrays = assert_within_eps(img, img_adv, target_eps, norm=norm)
            # print("projection check:", report)

        wts_and_bs = builder.build_wts_and_bs_from_img(img, WeightSpaceFeatures(*init_params))
        wsfeat = WeightSpaceFeatures(*wts_and_bs).to(ctx.device)
        clean_pred = victim(wsfeat).argmax(dim=1)
        clean_correct += (clean_pred == label).sum().item()

        # build inr from perturbed image 
        wts_and_bs_adv = builder.build_wts_and_bs_from_img(img_adv, WeightSpaceFeatures(*init_params))
        wsfeat = WeightSpaceFeatures(*wts_and_bs_adv).to(ctx.device)
        adv_pred = victim(wsfeat).argmax(dim=1)
        adv_correct += (adv_pred == label).sum().item()

        total += label.numel()
        print(f"Batch {i}: Clean Acc {(clean_correct/total)*100:.2f}%, Adv Acc {(adv_correct/total)*100:.2f}%")
        if i % 10 == 0:
            builder.visualize_reconstruction(WeightSpaceFeatures(*wts_and_bs), 
                                            WeightSpaceFeatures(*wts_and_bs_adv), 
                                            img, 
                                            img_adv, 
                                            eps=attack.eps,
                                            steps=attack.steps,
                                            norm=kwargs['norm'],
                                            plot_dir=ctx.exp_plot_path, 
                                            title=f"attack_{i}")

    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
    }


#TODO
def run_inr_perturbation_attack(
    ctx,
    eps: float | None = None,
    logging: bool = False,
    basemodel_ckpt_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Common pipeline for gaussian / transfer INR-space (weight-space) attacks.

    1. Build an INR+image DataLoader.
    2. Load victim model(s).
    3. For each batch: add perturbation to INR weights.
    4. Evaluate clean vs adversarial accuracy.

    Args:
        ctx: The :class:`Attacks` instance.
        eps: Perturbation strength (scales noise std for Gaussian, or L_inf bound
             for transfer).  Defaults to 0.3 for MNIST, 0.03 for CIFAR.
        logging: If True, save visualization plots.
        **kwargs: Additional parameters.

    Returns:
        Dict with ``clean_acc``, ``adv_acc``, ``total``, ``eps``, ``attack_space``.
    """
    if ctx.attack_type not in ("gaussian", "transfer"):
        raise ValueError(f"run_inr_perturbation_attack only supports gaussian/transfer, got: {ctx.attack_type}")

    # Default eps based on dataset
    if eps is None:
        eps = DEFAULT_EPS[ctx.dataset_name]

    # 1) Build loader over siren params + original images/labels
    inr_loader = ctx._build_original_and_inr_loader()

    # 2) Load victim model(s)
    # path_config is forwarded via **kwargs (added by Attacks.run())
    victim = load_victim_model(
        victim_model=ctx.victim_model,
        dataset_name=ctx.dataset_name,
        inr_config=ctx.inr_config,
        device=ctx.device,
        testloader=inr_loader,
        ckpt_path=basemodel_ckpt_path,
        **kwargs,
    )

    # Setup logging if enabled
    if logging:
        sidelen = 32 if ctx.dataset_name == "cifar" else 28
        c_out = 3 if ctx.dataset_name == "cifar" else 1
        builder = BatchedINRBuilder(
            sidelen=sidelen,
            hidden_features=32,
            hidden_layers=1,
            c_out=c_out,
            steps=5000,
            lr=5e-4,
            amp=True,
            device=ctx.device,
        )
        plot_dir = os.path.join(ctx.exp_plot_path, "inr_perturb")
        os.makedirs(plot_dir, exist_ok=True)

    clean_correct = 0
    adv_correct = 0
    total = 0

    # 3) Process each batch
    for i, (params, img, y) in tqdm(enumerate(inr_loader), desc=f"INR {ctx.attack_type} attack"):
        wsfeat = WeightSpaceFeatures(*params).to(ctx.device)
        y = y.to(ctx.device, non_blocking=True)

        # Generate adversarial INR weights
        if ctx.attack_type == "gaussian":
            # Add Gaussian noise directly to weights and biases
            adv_weights = tuple(
                w + torch.randn_like(w) * eps for w in wsfeat.weights
            )
            adv_biases = tuple(
                b + torch.randn_like(b) * eps for b in wsfeat.biases
            )
            adv_wsfeat = WeightSpaceFeatures(adv_weights, adv_biases)

        elif ctx.attack_type == "transfer":
            # Transfer attack in INR space: random perturbation with L_inf clipping
            delta_weights = tuple(
                torch.clamp(torch.randn_like(w) * eps, -eps, eps) for w in wsfeat.weights
            )
            delta_biases = tuple(
                torch.clamp(torch.randn_like(b) * eps, -eps, eps) for b in wsfeat.biases
            )
            adv_weights = tuple(w + dw for w, dw in zip(wsfeat.weights, delta_weights))
            adv_biases = tuple(b + db for b, db in zip(wsfeat.biases, delta_biases))
            adv_wsfeat = WeightSpaceFeatures(adv_weights, adv_biases)

        # 4) Evaluate clean and adversarial predictions
        with torch.no_grad():
            if ctx.victim_model == "nfn":
                model = victim["nfnet"]
                clean_logits = model(wsfeat)
                adv_logits = model(adv_wsfeat)

            elif ctx.victim_model == "nft":
                nfnet: AutoEncoder = victim["nfnet"]
                clf: nn.Module = victim["clf"]
                # INR -> embedding -> logits
                clean_emb = nfnet.encode(wsfeat)
                adv_emb = nfnet.encode(adv_wsfeat)
                clean_logits = clf(clean_emb.view(clean_emb.shape[0], 16, 256))
                adv_logits = clf(adv_emb.view(adv_emb.shape[0], 16, 256))

            elif ctx.victim_model == "dws":
                dws_model = victim["dws"]
                # Convert WeightSpaceFeatures to DWS format (transpose + feature dim)
                clean_weights_dws, clean_biases_dws = wsfeat_to_dws(wsfeat)
                clean_logits = dws_model((clean_weights_dws, clean_biases_dws))

                adv_weights_dws, adv_biases_dws = wsfeat_to_dws(adv_wsfeat)
                adv_logits = dws_model((adv_weights_dws, adv_biases_dws))

            elif ctx.victim_model == "scalegmn":
                # ScaleGMN requires graph conversion
                scalegmn_config = victim.get("config", {})
                batch_params = get_scalegmn_batch_params(scalegmn_config)

                clean_batch = wsfeat_to_pyg_batch(wsfeat, **batch_params)
                adv_batch = wsfeat_to_pyg_batch(adv_wsfeat, **batch_params)

                clean_batch = clean_batch.to(ctx.device)
                adv_batch = adv_batch.to(ctx.device)

                clean_logits = victim["scalegmn"](clean_batch)
                adv_logits = victim["scalegmn"](adv_batch)

            else:
                raise ValueError(f"Unsupported victim_model: {ctx.victim_model}")

            clean_pred = clean_logits.argmax(dim=1)
            adv_pred = adv_logits.argmax(dim=1)

            clean_correct += (clean_pred == y).sum().item()
            adv_correct += (adv_pred == y).sum().item()
            total += y.numel()

        # Visualization for debugging
        if logging and i % 10 == 0:
            adv_img = builder.build_img_from_wts_and_bs(
                wts_and_bs=adv_wsfeat.to("cpu"), if_wsfeat=True
            ).detach().cpu()
            img_clean = (img.permute(0, 2, 3, 1).detach().cpu() + 1.0) / 2.0

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for j in range(min(2, img_clean.shape[0])):
                axes[j, 0].imshow(img_clean[j])
                axes[j, 0].set_title("Clean image")
                axes[j, 1].imshow(adv_img[j])
                axes[j, 1].set_title(f"Recon from {ctx.attack_type} perturbed INR")
                axes[j, 0].axis("off")
                axes[j, 1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"batch_{i}.png"))
            plt.close(fig)

    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
        "total": total,
        "eps": eps,
        "attack_space": "inr",
        "attack_type": ctx.attack_type,
    }

def run_inr_pgd(
    dataset_name,
    device='cuda',
    **kwargs,
):
    victim = kwargs["victim_wrapped"]
    victim.eval()
    attack_emb = kwargs['attack_space'] == "embedding"

    _, _, testloader = kwargs['dataloader']
    builder = kwargs['builder']
    n_restarts = kwargs.get("n_restarts", 5)
    attack = get_attacker(victim, dataset_name, kwargs)

    clean_correct, adv_correct, total = 0, 0, 0

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = WeightSpaceFeatures(inrs[0], inrs[1]).to(device)

        # with torch.no_grad():
        #     matrices = victim._inr2vec_to_matrices(srcs)
        #     embedding = victim.model(matrices)
        #     print(f"embedding shape: {embedding.shape}")
        #     print(f"emb[0]: mean={embedding[0].mean():.4f}, std={embedding[0].std():.4f}")
        #     print(f"emb[1]: mean={embedding[1].mean():.4f}, std={embedding[1].std():.4f}")
        #     print(f"emb diff: {(embedding[0] - embedding[1]).abs().max():.6f}")
        B = srcs.weights[0].shape[0]

        # ---- multi-restart PGD ----
        best_adv_srcs = srcs.map(lambda x: x.clone())
        best_loss = torch.full((B,), -float('inf'), device=device)

        for _ in range(n_restarts):
            with torch.enable_grad():
                adv_srcs_r = attack(srcs, labels)

            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
                    victim(adv_srcs_r), labels
                )

            update_mask = per_sample_loss > best_loss  # (B,)

            if update_mask.any():
                new_weights = tuple(
                    torch.where(
                        update_mask.view(-1, *([1] * (w_best.dim() - 1))),
                        w_new, w_best,
                    )
                    for w_best, w_new in zip(best_adv_srcs.weights, adv_srcs_r.weights)
                )
                new_biases = tuple(
                    torch.where(
                        update_mask.view(-1, *([1] * (b_best.dim() - 1))),
                        b_new, b_best,
                    )
                    for b_best, b_new in zip(best_adv_srcs.biases, adv_srcs_r.biases)
                )
                best_adv_srcs = WeightSpaceFeatures(new_weights, new_biases)

            best_loss = torch.where(update_mask, per_sample_loss, best_loss)

        adv_srcs = best_adv_srcs

        # ---- eval ----
        with torch.no_grad():
            clean_pred = victim(srcs).argmax(dim=1)
            adv_pred = victim(adv_srcs).argmax(dim=1)

            clean_correct += (clean_pred == labels).sum().item()
            adv_correct += (adv_pred == labels).sum().item()
            total += labels.numel()

        if (i % 10 == 1) and not attack_emb:
            builder.visualize_reconstruction(
                wts_and_bs=srcs,
                wts_and_bs_adv=adv_srcs,
                x=imgs, 
                eps=attack.eps, steps=attack.steps,
                norm=kwargs.get('norm', 'linf'),
                plot_dir=kwargs['exp_plot_path'],
                object="Perturbed",
                title=f"attack_{i}",
            )

    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
        "total": total,
    }

#TODO
def run_embedding_pgd(
    dataset_name,
    device='cuda', 
    save_intermediate_results=False,
    **kwargs):

    victim = kwargs["victim_wrapped"]
    victim.eval()

    # Use dataset-specific default eps_std if not provided
    eps_std = DEFAULT_EPS_STD.get(dataset_name, 0.1)
    print(f"Using default eps_std={eps_std} for dataset '{dataset_name}'")

    _, _, testloader = kwargs['dataloader']

    print("Computing embedding statistics...")
    emb_stats = compute_embedding_stats(testloader)
    print(f"Embedding stats - mean: {emb_stats['mean']:.4f}, std: {emb_stats['std']:.4f}, "
          f"min: {emb_stats['min']:.4f}, max: {emb_stats['max']:.4f}")
    
    n_restarts = kwargs.get("n_restarts", 5)
    attack = PGDEmbedding(
        model=victim,
        eps_std=eps_std,
        steps=kwargs.get("steps", 20),
        emb_std=emb_stats['std'],
    )

    clean_correct = 0
    adv_correct = 0
    total = 0

    print(f"Running embedding PGD attack with eps_std={eps_std}, steps={kwargs['steps']}")
    for i, batch in tqdm(enumerate(testloader)):
        _, _, _, embs, labels = batch

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            embs = embs.to(device, non_blocking=True)

        best_adv_embs = embs.clone()
        best_loss = torch.full((embs.shape[0],), -float('inf'), device=device)
        expand_shape = [-1] + [1] * (embs.dim() - 1)

        for _ in range(n_restarts):
            with torch.enable_grad():
                adv_embs_r = attack(embs, labels)
            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
                    victim(adv_embs_r), labels
                )
            update_mask = per_sample_loss > best_loss
            best_adv_embs = torch.where(update_mask.view(*expand_shape), adv_embs_r, best_adv_embs)
            best_loss = torch.where(update_mask, per_sample_loss, best_loss)


        with torch.no_grad():
            clean_logits = victim(embs)
            adv_logits = victim(best_adv_embs)

            clean_pred = clean_logits.argmax(dim=1)
            adv_pred = adv_logits.argmax(dim=1)

            clean_correct += (clean_pred == labels).sum().item()
            adv_correct += (adv_pred == labels).sum().item()
            total += labels.numel()

    clean_acc = clean_correct / max(total, 1)
    adv_acc = adv_correct / max(total, 1)

    print(f"Results - Clean Acc: {clean_acc:.4f}, Adv Acc: {adv_acc:.4f}")

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "total": total,
    }
