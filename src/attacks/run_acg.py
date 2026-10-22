"""
ACG (Auto Conjugate Gradient, ICML 2022) evaluation.
Same interface as run_pgd / run_deepfool_against_surrogate.
"""

import torch
from tqdm import tqdm

from attacks.torchattacks.attacks.acg import AUTOConjugate


def run_acg_against_surrogate(
        dataset_name,
        device='cuda',
        save_intermediate_results=False,
        **kwargs):
    """
    ACG attack evaluated on surrogate pipeline.
    Same interface as run_deepfool_against_surrogate.
    """

    # Load victim & surrogate
    victim = kwargs["victim_wrapped"]
    victim.eval()
    _, _, testloader = kwargs['dataloader']
    surrogate = kwargs['surrogate']
    surrogate.eval()

    # ACG params
    eps = kwargs.get("eps", 0.3)
    steps = kwargs.get("acg_steps", 100)
    num_restarts = kwargs.get("n_restarts", 1)
    img_scale = kwargs.get("img_scale", [-1, 1])
    criterion = kwargs.get("acg_criterion", "dlr")
    beta_method = kwargs.get("acg_beta", "HS")
    momentum_alpha = kwargs.get("acg_momentum", 0.75)
    eot_iter = kwargs.get("acg_eot", 1)
    thr_decr = kwargs.get("acg_rho", 0.75)
    initial_point = kwargs.get("acg_initial_point", "input")
    restart_point = kwargs.get("acg_restart_point", "random")

    attack = AUTOConjugate(
        model=surrogate,
        eps=eps,
        steps=steps,
        num_restarts=num_restarts,
        img_scale=img_scale,
        criterion=criterion,
        beta_method=beta_method,
        momentum_alpha=momentum_alpha,
        eot_iter=eot_iter,
        thr_decr=thr_decr,
        initial_point=initial_point,
        restart_point=restart_point,
    )

    clean_correct, adv_correct, total = 0, 0, 0

    if save_intermediate_results:
        res = []

    print(f"Running ACG against surrogate: eps={eps}, steps={steps}, "
          f"restarts={num_restarts}, criterion={criterion}, beta={beta_method}")

    for i, batch in tqdm(enumerate(testloader)):
        if len(batch) == 5:
            inrs, init_params, imgs, embs, labels = batch
        else:
            inrs, init_params, imgs, labels = batch

        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        with torch.enable_grad():
            adv_srcs = attack(srcs, labels)

        with torch.no_grad():
            clean_pred = surrogate(srcs).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = surrogate(adv_srcs).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

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