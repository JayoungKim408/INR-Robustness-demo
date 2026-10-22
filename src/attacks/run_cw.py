import torch
from tqdm import tqdm

# from attacks.torchattacks.attacks.cw import CW
from attacks.torchattacks.attacks.cw_ import CarliniWagnerL2Attack as CW


def run_cw_against_surrogate(
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

    # n_restarts = kwargs.get("n_restarts", 1)
    # torchattack implementation of CW attack
    # attack = CW(model=surrogate, 
    #             c=kwargs.get("c", 1), 
    #             kappa=kwargs.get("kappa", 0), 
    #             steps=kwargs.get("cw_steps", 50), 
    #             lr=kwargs.get("cw_lr", 0.01), 
    #             img_scale=[-1, 1]
                # )

    attack = CW(
                predict=surrogate, 
                clip_min=-1., 
                clip_max=1.,
                max_iterations=1000
                )
    
    clean_correct, adv_correct, total = 0, 0, 0

    # torch.cuda.reset_peak_memory_stats()
    print(f"Running CW attack against surrogate on {dataset_name} with max_iterations={attack.max_iterations}", flush=True)
    for i, batch in tqdm(enumerate(testloader)):
        inrs, init_params, imgs, labels = batch
        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            srcs = imgs.to(device, non_blocking=True)

        # Multi-restart PGD: run n_restarts times, keep per-sample worst case
        # best_adv_srcs = srcs.clone()
        # best_loss = torch.full((srcs.shape[0],), -float('inf'), device=device)
        # expand_shape = [-1] + [1] * (srcs.dim() - 1)

        # for _ in range(n_restarts):
        with torch.enable_grad():
            adv_srcs = attack(srcs, labels)
            # with torch.no_grad():
            #     per_sample_loss = nn.CrossEntropyLoss(reduction='none')(
            #         surrogate(adv_srcs_r), labels
            #     )
            # update_mask = per_sample_loss > best_loss
            # best_adv_srcs = torch.where(update_mask.view(*expand_shape), adv_srcs_r, best_adv_srcs)
            # best_loss = torch.where(update_mask, per_sample_loss, best_loss)

        # adv_srcs = best_adv_srcs
        # L2 norm (per sample)
        delta = adv_srcs - srcs
        l2_norms = delta.view(delta.shape[0], -1).norm(p=2, dim=1)
        linf_norms = delta.view(delta.shape[0], -1).abs().max(dim=1)[0]

        # 통계 출력
        print(f"L2:  mean={l2_norms.mean():.4f}, median={l2_norms.median():.4f}, max={l2_norms.max():.4f}", flush=True)
        print(f"Linf: mean={linf_norms.mean():.4f}, median={linf_norms.median():.4f}, max={linf_norms.max():.4f}", flush=True)

        with torch.no_grad():
            clean_pred = surrogate(srcs).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            adv_pred = surrogate(adv_srcs).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

        print(
            f"BATCH {i} | Clean acc: {clean_correct / max(total, 1):.4f} | "
            f"Adv acc against surrogate: {adv_correct / max(total, 1):.4f} ({adv_correct})", flush=True
        )
        # if save_intermediate_results:
        #     res.append({"clean_img": srcs.cpu(), "adv_img": adv_srcs.cpu(), "label": labels.cpu()})
        # print(f"Surrogate: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        # return 

    final_clean_acc = clean_correct / max(total, 1)
    final_adv_acc = adv_correct / max(total, 1)
    print(f"Final Clean Acc: {final_clean_acc:.4f}, Final Adv Acc against surrogate: {final_adv_acc:.4f}")
    return {
        "surrogate_clean_acc": final_clean_acc,
        "surrogate_adv_acc": final_adv_acc,
    }

