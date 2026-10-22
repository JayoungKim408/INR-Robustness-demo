"""
MALT (NeurIPS 2024) attack evaluation.
Uses Jacobian-normalized target selection + targeted PGD.
Same interface as run_pgd / run_deepfool_against_surrogate.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from attacks.torchattacks.attacks.pgd import PGD


# =========================================================
# MALT core functions (adapted from Melamed et al., 2024)
# =========================================================

def to_one_hot(y, num_classes=10):
    y = y.detach().clone().view(-1, 1)
    return y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)


def fast_find_matrix_around_image(images, targets, inxs, model, num_classes=1000, batch_size=50):
    images_repeated = torch.repeat_interleave(images.detach().clone().cpu(), inxs.shape[1], dim=0).detach()

    eye = to_one_hot(inxs, num_classes=num_classes)
    if targets is not None:
        targets_repeated = torch.repeat_interleave(targets, inxs.shape[1], dim=0).detach()
        eye = eye.scatter(dim=1, index=targets_repeated.unsqueeze(1), value=-1)
    eye = eye.type(torch.FloatTensor)

    resmatrices = batch_gradient_w_loss(model, images_repeated, eye, batch_size=batch_size)
    resmatrices = resmatrices.reshape((images.shape[0], inxs.shape[-1],
                                        images.shape[-3] * images.shape[-2] * images.shape[-1]))

    return resmatrices


def batch_gradient_w_loss(model, input, backward_var, batch_size=50):
    grads = torch.tensor([])
    start = 0
    while start < input.shape[0]:
        end = min(start + batch_size, input.shape[0])
        batch_inputs = input[start:end].cuda()
        batch_backward_var = backward_var[start:end].cuda()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            batch_inputs = nn.Parameter(batch_inputs)
            batch_inputs.requires_grad_(True)
            output = model(batch_inputs)
            output.backward(batch_backward_var)
            g = batch_inputs.grad.data.detach().clone().cpu()
            grads = torch.cat((grads, g), dim=0)
            batch_inputs.grad.zero_()
            batch_inputs.detach()

        start = end

    return grads


def get_class_rank(x, next_topk_inxs, ord, linmatrix, srcoutput, srcclass, type=torch.float):
    epsilon_matrix = torch.zeros((x.shape[0], next_topk_inxs.shape[-1]), dtype=type)

    src_classes_scores = srcoutput[range(srcoutput.shape[0]), srcclass.view(-1)].unsqueeze(1)
    score_diffs = torch.repeat_interleave(src_classes_scores, next_topk_inxs.shape[-1], dim=1) - \
                  srcoutput[torch.arange(srcoutput.shape[0]).unsqueeze(-1), next_topk_inxs]
    matrix_raw_diffs = linmatrix#.cuda()

    if ord == 2:
        matrix_raw_diff_norms = torch.norm(matrix_raw_diffs, p=2, dim=2)

    if ord == np.inf:
        matrix_raw_diff_norms = torch.norm(matrix_raw_diffs, p=1, dim=2)

    non_zero_row_mask = (matrix_raw_diff_norms != 0)

    if non_zero_row_mask.sum() != 0:
        epsilon_matrix[non_zero_row_mask] = \
            score_diffs[non_zero_row_mask].cpu() / ((matrix_raw_diff_norms[non_zero_row_mask]).cpu())

    all_classes_matrix = np.inf * torch.ones(x.shape[0], srcoutput.shape[-1])
    all_classes_matrix[torch.arange(x.shape[0]).unsqueeze(-1), next_topk_inxs] = epsilon_matrix
    return all_classes_matrix


def get_k_best_target_classes(model, data, target, ord, topk, testk=None, batch_size=50):
    with torch.no_grad():
        srcoutput = model(data)
    _, srcclass = torch.max(srcoutput, dim=1)
    n_srcoutput = srcoutput.scatter(dim=1, index=target.unsqueeze(1), value=-float("inf"))
    if testk == None or testk > srcoutput.shape[-1]-1:
        testk = srcoutput.shape[-1]-1

    _, next_topk_inxs = torch.topk(n_srcoutput, k=testk, dim=1)
    if topk > srcoutput.shape[-1]-1:
        topk = srcoutput.shape[-1]-1

    linmatrix = fast_find_matrix_around_image(data, target, next_topk_inxs, model, num_classes=srcoutput.shape[-1], batch_size=batch_size)
    epsilon_matrix = get_class_rank(data, next_topk_inxs, ord, linmatrix, srcoutput, srcclass, type=torch.float)
    return epsilon_matrix.topk(dim=1, largest=False, k=topk)[1].cuda()



def malt_attack_batch(attack_func, model, X, y, topk=9, testk=9, norm=np.inf, batch_size=50):
    """Run MALT: try targets in order of Jacobian-normalized ranking."""
    device = X.device
    adv_total = X.detach().clone()

    ebt_targets = get_k_best_target_classes(
        model=model, data=X.detach(), target=y.detach(),
        ord=norm, topk=topk, testk=testk, batch_size=batch_size
    )

    still_failed = torch.ones(X.shape[0], dtype=torch.bool, device=device)

    for i in range(ebt_targets.shape[1]):
        if not still_failed.any():
            break

        adv_target = ebt_targets[:, i]
        current_to_attack = still_failed.clone()

        X_adv = attack_func(
            model,
            X[current_to_attack].detach(),
            adv_target[current_to_attack].detach()
        ).detach()

        with torch.no_grad():
            preds = model(X_adv).argmax(dim=1)
        succeeded = (preds == adv_target[current_to_attack])

        update_mask = torch.zeros_like(still_failed)
        update_mask[current_to_attack] = succeeded
        adv_total[update_mask] = X_adv[succeeded]
        still_failed[update_mask] = False

    return adv_total, still_failed


# =========================================================
# Evaluation functions
# =========================================================

def run_malt_against_surrogate(
        dataset_name,
        device='cuda',
        save_intermediate_results=False,
        **kwargs):
    """
    MALT attack evaluated on surrogate pipeline.
    Same interface as run_deepfool_against_surrogate.
    """
    from attacks.utils import get_dataset_params, DEFAULT_EPS

    # Load victim & surrogate
    victim = kwargs["victim_wrapped"]
    victim.eval()
    _, _, testloader = kwargs['dataloader']
    surrogate = kwargs['surrogate']
    surrogate.eval()

    # Attack params
    ds = get_dataset_params(dataset_name)
    eps = kwargs.get("eps", DEFAULT_EPS.get(dataset_name, 0.3))
    alpha = kwargs.get("alpha", eps / 10)
    steps = kwargs.get("steps", 20)
    img_scale = kwargs.get("img_scale", [-1, 1])
    n_restarts = kwargs.get("n_restarts", 1)
    topk = kwargs.get("malt_topk", ds["num_classes"] - 1)
    testk = kwargs.get("malt_testk", ds["num_classes"] - 1)
    malt_batch_size = kwargs.get("malt_batch_size", 50)

    # Targeted PGD as base attack
    def targeted_pgd(model, data, target):
        attack = PGD(model=model, eps=eps, alpha=alpha, steps=steps,
                     img_scale=img_scale, random_start=True)
        attack.set_mode_targeted_by_label()
        return attack(data, target)

    clean_correct, adv_correct, total = 0, 0, 0

    if save_intermediate_results:
        res = []

    print(f"Running MALT against surrogate: eps={eps}, alpha={alpha}, steps={steps}, "
          f"topk={topk}, testk={testk}, restarts={n_restarts}")

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        # Multi-restart: keep best adversarial per sample
        best_adv = srcs.clone()
        best_loss = torch.full((srcs.shape[0],), -float('inf'), device=device)
        expand_shape = [-1] + [1] * (srcs.dim() - 1)

        for r in range(n_restarts):
            with torch.enable_grad():
                adv_srcs, still_failed = malt_attack_batch(
                    attack_func=targeted_pgd,
                    model=surrogate,
                    X=srcs,
                    y=labels,
                    topk=topk,
                    testk=testk,
                    norm=np.inf,
                    batch_size=malt_batch_size,
                )
            print(f"MALT still_failed: {still_failed.sum().item()}/{srcs.shape[0]}")

            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
                    surrogate(adv_srcs), labels
                )
            update_mask = per_sample_loss > best_loss
            best_adv = torch.where(update_mask.view(*expand_shape), adv_srcs, best_adv)
            best_loss = torch.where(update_mask, per_sample_loss, best_loss)

        adv_srcs = best_adv

        with torch.no_grad():
            clean_pred = surrogate(srcs).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = surrogate(adv_srcs).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()
            
        l2_diff = (adv_srcs - srcs).reshape(srcs.shape[0], -1).norm(dim=1)
        print(f"Mean L2 perturbation: {l2_diff.mean().item():.4f}")
        print(f"Samples with zero perturbation: {(l2_diff == 0).sum().item()}/{srcs.shape[0]}")

        print(
            f"BATCH {i} | Clean acc: {clean_correct / max(total, 1):.4f} | "
            f"Adv acc against surrogate: {adv_correct / max(total, 1):.4f} ({adv_correct})"
        , flush=True)

        if save_intermediate_results:
            res.append({"clean_img": srcs.cpu(), "adv_img": adv_srcs.cpu(), "label": labels.cpu()})

    final_clean_acc = clean_correct / max(total, 1)
    final_adv_acc = adv_correct / max(total, 1)
    print(f"Final Clean Acc: {final_clean_acc:.4f}, Final Adv Acc against surrogate: {final_adv_acc:.4f}")
    return {
        "surrogate_clean_acc": final_clean_acc,
        "surrogate_adv_acc": final_adv_acc,
    }


def run_malt(
        dataset_name,
        device='cuda',
        save_intermediate_results=False,
        **kwargs):
    """
    MALT attack evaluated on real pipeline (via builder).
    Same interface as run_pgd.
    """
    from attacks.utils import get_dataset_params, DEFAULT_EPS
    from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures

    # Load victim & surrogate
    victim = kwargs["victim_wrapped"]
    victim.eval()
    _, _, testloader = kwargs['dataloader']
    surrogate = kwargs['surrogate']
    surrogate.eval()
    builder = kwargs['builder']
    is_3d = dataset_name == "shapenet"

    # Attack params
    ds = get_dataset_params(dataset_name)
    eps = kwargs.get("eps", DEFAULT_EPS.get(dataset_name, 0.3))
    alpha = kwargs.get("alpha", eps / 10)
    steps = kwargs.get("steps", 20)
    img_scale = kwargs.get("img_scale", [-1, 1])
    n_restarts = kwargs.get("n_restarts", 1)
    topk = kwargs.get("malt_topk", ds["num_classes"] - 1)
    testk = kwargs.get("malt_testk", ds["num_classes"] - 1)
    malt_batch_size = kwargs.get("malt_batch_size", 50)

    def targeted_pgd(model, data, target):
        attack = PGD(model=model, eps=eps, alpha=alpha, steps=steps,
                     img_scale=img_scale, random_start=True)
        attack.set_mode_targeted_by_label()
        return attack(data, target)

    def build_image(init_params, imgs):
        wts_and_bs = builder.build_wts_and_bs_from_img(imgs, WeightSpaceFeatures(*init_params))
        return WeightSpaceFeatures(*wts_and_bs).to(device)

    clean_correct, adv_correct, total = 0, 0, 0

    print(f"Running MALT: eps={eps}, alpha={alpha}, steps={steps}, "
          f"topk={topk}, testk={testk}, restarts={n_restarts}")

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        # Multi-restart
        best_adv = srcs.clone()
        best_loss = torch.full((srcs.shape[0],), -float('inf'), device=device)
        expand_shape = [-1] + [1] * (srcs.dim() - 1)

        for r in range(n_restarts):
            with torch.enable_grad():
                adv_srcs, still_failed = malt_attack_batch(
                    attack_func=targeted_pgd,
                    model=surrogate,
                    X=srcs,
                    y=labels,
                    topk=topk,
                    testk=testk,
                    norm=np.inf,
                    batch_size=malt_batch_size,
                )

            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
                    surrogate(adv_srcs), labels
                )
            update_mask = per_sample_loss > best_loss
            best_adv = torch.where(update_mask.view(*expand_shape), adv_srcs, best_adv)
            best_loss = torch.where(update_mask, per_sample_loss, best_loss)

        adv_srcs = best_adv

        with torch.no_grad():
            if is_3d:
                inp = WeightSpaceFeatures(*inrs).to(device)
                adv_inp, _ = builder.build_wts_and_bs_from_pcd(adv_srcs, WeightSpaceFeatures(*init_params))
                adv_inp = WeightSpaceFeatures(*adv_inp).to(device)
            else:
                inp = build_image(init_params, srcs)
                adv_inp = build_image(init_params, adv_srcs)

            clean_pred = victim(inp).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = victim(adv_inp).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

        print(
            f"BATCH {i} | Clean acc: {clean_correct / max(total, 1):.4f} | "
            f"Adv acc: {adv_correct / max(total, 1):.4f} ({adv_correct})"
        , flush=True)

    final_clean_acc = clean_correct / max(total, 1)
    final_adv_acc = adv_correct / max(total, 1)
    print(f"Final Clean Acc: {final_clean_acc:.4f}, Final Adv Acc: {final_adv_acc:.4f}")
    return {
        "clean_acc": final_clean_acc,
        "adv_acc": final_adv_acc,
        "total": total,
    }