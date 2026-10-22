from math import e
import os
import math
# import hydra
import torch
from torch import nn
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import default_collate
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional

from utils.wandb_manager import WandbManager
from utils.path_config import PathConfig

from ..utils import BatchedINRBuilder
from attacks.torchattacks.attacks.pgd import PGD
from attacks.torchattacks.attacks.pgdl1 import PGDL1
from attacks.torchattacks.attacks.pgdl2 import PGDL2

from .train_hypernet.meta_modules import ConvolutionalNeuralProcessImplicit2DHypernet
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures, state_dict_to_tensors

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=512, hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
        layers += [nn.Linear(hidden_features, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def convert_to_state_dicts(hypo_params):
    """Convert hypo_params into list of state dicts for individual models"""
    state_dicts = []
    batch_size = next(iter(hypo_params.values())).shape[0]
    
    for i in range(batch_size):
        state_dict = {}
        for key, value in hypo_params.items():
            # Remove batch dimension but KEEP gradient connection (no detach/clone)
            state_dict[key] = value[i]
        state_dicts.append(state_dict)
        
    return state_dicts

class SurrogateClassifier(nn.Module):
    """Image -> Embedding surrogate -> Classifier (logits)."""
    def __init__(self, surrogate: nn.Module, victim_wrapped: nn.Module):
        super().__init__()
        self.surrogate = surrogate
        self.victim_wrapped = victim_wrapped

    def forward(self, x, return_wsfeats=False):
        inr, _ = self.surrogate.get_hypo_net_weights({'img_sparse': x})
        hypo_params_state_dicts = convert_to_state_dicts(inr)
        wts_and_bs = [state_dict_to_tensors(sd) for sd in hypo_params_state_dicts]
        wts_and_bs = default_collate(wts_and_bs)
        wsfeats = WeightSpaceFeatures(*wts_and_bs).to(x.device)

        logits = self.victim_wrapped(wsfeats)
        if return_wsfeats:
            return logits, wsfeats
        else:
            return logits

def main(surrogate=None,
         surrogate_ckpt=None,
         dataset_name=None,
         victim=None,
         device='cuda',
         wandb_manager: Optional[WandbManager] = None,
         base_path=None,
         **kwargs,
):
    """
    Run hypernet attack (i2i-hypernet).

    Additional kwargs for INR ablation studies:
        siren_path: Optional custom siren path
        inr_hidden_layers: Optional INR hidden layers
        inr_hidden_features: Optional INR hidden features
        inr_first_omega_0: Optional INR first omega_0
        inr_config: Optional INRConfig object
    """
    # Extract optional INR config from kwargs
    siren_path = kwargs.pop("siren_path", None)
    inr_config = kwargs.pop("inr_config", None)
    inr_hidden_layers = kwargs.pop("inr_hidden_layers", 1)
    inr_hidden_features = kwargs.pop("inr_hidden_features", 32)
    inr_first_omega_0 = kwargs.pop("inr_first_omega_0", 30.0)
    inr_hidden_omega_0 = kwargs.pop("inr_hidden_omega_0", 30.0)


    # Victim model must be passed from run_attacks.py
    if victim is None:
        raise ValueError("victim must be provided. Load victim model in run_attacks.py and pass it here.")
    victim.eval()

    if siren_path is None:
        path_config = PathConfig(base_path=base_path)
        siren_path = path_config.get_siren_dir(dataset=dataset_name, inr_config=inr_config)

    print(f"Using SIREN path: {siren_path}")
    print(f"INR architecture: depth={inr_hidden_layers}, width={inr_hidden_features}, scale={inr_first_omega_0}")

    # Dataset configuration
    data_path = os.path.join(base_path, "raw")
    split_points = {"mnist": [55000, 60000], "fashion": [55000, 60000], "cifar": [45000, 50000]}
    dataset_split_points = split_points.get(dataset_name, [55000, 60000])

    from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset
    testset = get_SirenAdnOriginalDataset(
        siren_path=siren_path,
        siren_prefix="randinit_smaller",
        data_path=data_path,
        split="test",
        split_points=dataset_split_points,
    )
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8, drop_last=False)

    # surrogate model
    in_ch = 3 if dataset_name == "cifar" else 1
    side = 32 if dataset_name == "cifar" else 28
    emb_shape = (16, 256)

    if surrogate is None:
        surrogate = ConvolutionalNeuralProcessImplicit2DHypernet(in_features=in_ch,
                                                                 out_features=in_ch,
                                                                 image_resolution=(side, side),
                                                                 hidden_features=inr_hidden_features,
                                                                 num_hidden_layers=inr_hidden_layers
                                                                 ).to(device)


        # surrogate_ckpt = path_config.get_surrogate_checkpoint_path(
        #     attack_type="i2i-hypernet",
        #     dataset=dataset_name,
        #     victim_model=victim_model,
        # )
        if os.path.exists(surrogate_ckpt):
            surrogate.load_state_dict(torch.load(surrogate_ckpt, map_location="cpu", weights_only=False))
    surrogate.eval()
    # hid = surrogate.encoder(img)
    # inr = surrogate.hyper_net(hid) -> nfn here
    # embedding = surrogate.hypo_net(img, inr) 

    # load inr2array model (NFN encoder)
    # inr2array_cfg = OmegaConf.load(path_config.get_hydra_config(model=victim_model, dataset=dataset_name, subdir="inr2array", inr_config=inr_config))
    # nfnet_state = torch.load(path_config.get_inr2array_checkpoint(model=victim_model, dataset=dataset_name, inr_config=inr_config), map_location="cpu", weights_only=False)

    # Build NFN spec using a small sample from the data and current surrogate
    # with torch.no_grad():
    #     # return params, img, embedding, data_label

    #     x_sample = next(iter(testloader))[1].to(device)
    #     emb_sample = surrogate.encoder(x_sample)
    #     params_sample = surrogate.hyper_net(emb_sample)
    #     sd_list = convert_to_state_dicts(params_sample)
    #     wnb = [state_dict_to_tensors(sd) for sd in sd_list]
    #     wnb = default_collate(wnb)
    #     wsfeat_sample = WeightSpaceFeatures(*wnb).to("cpu")
    #     spec = network_spec_from_wsfeat(wsfeat_sample, set_all_dims=True)

    # nfnet = hydra.utils.instantiate(inr2array_cfg.model, spec, dset_data_type=dataset_name, compile=False).to(device)
    # nfnet.load_state_dict(nfnet_state)
    # nfnet.eval()

    # # for p in nfnet.parameters():
    # #     p.requires_grad = False

    # clf_state = torch.load(path_config.get_classifier_checkpoint(model=victim_model, dataset=dataset_name, inr_config=inr_config), map_location="cpu", weights_only=False)
    # clf = TransformerClassifier().to(device)
    # clf.load_state_dict(clf_state)
    # clf.eval()

    # Note: SurrogateClassifier takes (surrogate, classifier, nfnet)
    model = SurrogateClassifier(surrogate, victim.clf, victim.nfnet).to(device)

    # evaluation
    clean_correct, adv_correct, total = 0, 0, 0

    if kwargs.get("eps") is not None:
        eps = kwargs["eps"]
    else:
        eps = 0.3 if dataset_name == "mnist" else 0.03
    steps = kwargs.get("steps", 20)
    norm = kwargs.get("norm", "linf")
    PGD_MAP = {"linf": PGD,
               "l2": PGDL2,
               "l1": PGDL1}
        
    # If dataset is normalized differently (e.g., CIFAR), consider adapting eps/clamp accordingly.
    eps_scaled = eps * 2 # for [-1, 1]
    
    if norm == 'l2':
        eps = math.sqrt(side * side * in_ch) * eps_scaled
    elif norm == 'l1':
        eps = side * side * in_ch * eps_scaled
    elif norm == 'linf':
        eps = eps_scaled

    gamma = 2 * eps / steps
    builder = BatchedINRBuilder(batch_size=64, 
                                sidelen=side, 
                                hidden_features=inr_hidden_features, 
                                hidden_layers=inr_hidden_layers, 
                                first_omega_0=inr_first_omega_0,
                                hidden_omega_0=inr_hidden_omega_0,
                                c_out=in_ch, 
                                steps=5000, 
                                lr=5e-4, 
                                amp=True, 
                                device=device)
    
    attack = PGD_MAP[norm](model=model, 
                            eps=eps, 
                            alpha=gamma,
                            steps=steps,
                            img_scale=[-1, 1]
                            )
    
    print(f"PGD attack initialized with eps={eps}, alpha={gamma}, steps={steps}, norm={norm}")
    for i, (inr, x, y) in tqdm(enumerate(testloader)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits_clean, wsfeats_clean = model(x, return_wts_and_bs=True)
        pred_clean = logits_clean.argmax(dim=1)
        clean_correct += (pred_clean == y).sum().item()

        with torch.enable_grad():
            x_adv = attack(x, y)

        # pred_eps_linf = infer_eps_linf(x, x_adv, input_scale='-1,1')['eps_linf_max_pixels']
        # print(f"Predicted eps_linf: {pred_eps_linf}")
        report, _, _ = assert_within_eps(x, x_adv, eps_scaled, norm=norm)
        print(report)

        logits_adv, wsfeats_adv = model(x_adv, return_wts_and_bs=True)
        if i % 10 == 0:
            # builder.visualize_reconstruction(params_adv, x, x_adv, buffers={}, title=f"recon_hyper_perturbed_{dataset_name}_{i})")
            x_adv_img = builder.build_img_from_wts_and_bs(wsfeats_adv, if_wsfeat=True)
            x_img = builder.build_img_from_wts_and_bs(wsfeats_clean, if_wsfeat=True)

            x = (x + 1.0) / 2.0
            fig, axes = plt.subplots(4, 2, figsize=(6, 4 * 2))
            for j in range(4):
                axes[j, 0].imshow(x_img[j].detach().cpu().numpy())
                # axes[j, 1].imshow(x_adv_img[j].permute(1, 2, 0).detach().cpu().numpy())
                axes[j, 1].imshow(x_adv_img[j].detach().cpu().numpy())

                axes[j, 0].set_title(f"Clean Original")
                axes[j, 1].set_title("Perturbed Original")

                axes[j, 0].axis("off"), axes[j, 1].axis("off")

            plt.tight_layout()
            exp_plot_path = kwargs.get("exp_plot_path", ".")
            os.makedirs(exp_plot_path, exist_ok=True)
            plt.savefig(os.path.join(exp_plot_path, f"hypernet_attack_{dataset_name}_{i}.png"))
            if wandb_manager and wandb_manager.is_active:
                wandb_manager.log_image("images", plt)
            plt.close(fig)

        pred_adv = logits_adv.argmax(dim=1)
        adv_correct += (pred_adv == y).sum().item()

        total += y.numel()

    print(f"Hypernet surrogate — Dataset={dataset_name} | Clean acc: {clean_correct / max(total,1):.4f} | Adv acc: {adv_correct / max(total,1):.4f}")
    return {"clean_acc": clean_correct/ max(total,1), "adv_acc": adv_correct/ max(total,1), "total": total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mnist")
    parser.add_argument("--victim_model", type=str, default="nft")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    # Initialize wandb manager for standalone execution (project defaults to INR-Robustness-exp)
    wandb_manager = WandbManager(enabled=args.wandb)
    if args.wandb:
        run_name = wandb_manager.generate_run_name(
            dataset=args.dataset_name,
            model=args.victim_model,
            attack_type="hypernet",
        )
        wandb_manager.init_run(
            name=run_name,
            group=f"hypernet_attack_{args.dataset_name}",
            tags=wandb_manager.generate_tags(
                dataset=args.dataset_name,
                model=args.victim_model,
                attack_type="hypernet",
            ),
            config=dict(
                dataset=args.dataset_name,
                victim_model=args.victim_model,
            ),
            job_type="attack",
        )

    try:
        _ = main(
            dataset_name=args.dataset_name,
            victim_model=args.victim_model,
            wandb_manager=wandb_manager,
        )
    finally:
        wandb_manager.finish_run()

    # This attack is available for NFN, NFT, and ScaleGMN.