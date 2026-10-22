import os

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import default_collate
import argparse
from torch.nn.parameter import Parameter
from torch.amp import autocast
# from attacks.gaussian_perturb.gaussian_attack import infer_eps_linf
from .utils import load_data
from basemodels.utils.mwt.meta_sgd import MetaSGD
from basemodels.utils.mwt.wt import WT
from basemodels.utils.mwt.siren_model import SirenModel
from basemodels.utils.mwt.tconfig import get_config
from basemodels.utils.mwt.train import build_classifier, forward

from attacks.torchattacks.attacks.pgd import PGD
from attacks.torchattacks.attacks.pgdl1 import PGDL1
from attacks.torchattacks.attacks.pgdl2 import PGDL2

import matplotlib.pyplot as plt

class MWT_Pipeline(nn.Module):
    def __init__(self, conf, siren_rgb, classifier, meta_sgd, device, PI):
        super(MWT_Pipeline, self).__init__()
        self.conf = conf
        self.device = device
        self.siren_rgb = siren_rgb
        self.classifier = classifier
        self.meta_sgd = meta_sgd
        self.PI = PI
        
    def forward(self, x):
        preview_dict = {}
        with autocast(device_type=self.device, dtype=torch.float16, enabled=self.conf['auto_cast']):
            logits, inner_loss, psnr_running = forward(self.conf, 
                                                        x, 
                                                        preview_dict,
                                                        build_graph=True,
                                                        device=self.device,
                                                        PI=self.PI,
                                                        siren_rgb=self.siren_rgb,
                                                        classifier=self.classifier,
                                                        meta_sgd=self.meta_sgd,
                                                        eval_mode=True,
                                                        )
        return logits

def img_side_by_side(x, x_adv):
    x = x + 1 / 2
    x_adv = x_adv + 1 / 2
    x = x.clamp(0, 1)
    x_adv = x_adv.clamp(0, 1)
    x = x.detach().cpu().numpy()
    x_adv = x_adv.detach().cpu().numpy()

    x = x.transpose(0, 2, 3, 1)
    x_adv = x_adv.transpose(0, 2, 3, 1)

    # fig, axes = plt.subplots(4, 2, figsize=(4, 2))
    fig, axes = plt.subplots(4, 2, figsize=(6, 4 * 2))

    for i in range(4):
        axes[i, 0].imshow(x[i])
        axes[i, 1].imshow(x_adv[i])
        axes[i, 0].axis("off"), axes[i, 1].axis("off")
        axes[i, 0].set_title("Clean")
        axes[i, 1].set_title("Perturbed")
    plt.tight_layout()
    plt.savefig(f"plots/mwt_e2e_attack_img_side_by_side.png")
    plt.close()

def main(dataset_name=None, 
         base_path=None,
         **kwargs):

    conf = get_config()
    device = conf['device']

    if conf['inner_steps_test'] is None:
        conf['inner_steps_test'] = conf['inner_steps']

    # convert to a dict
    conf['siren_dim'] = [ conf['dim'] ] * conf['siren_depth']
    conf['nfn_channels'] = [conf['dim']] * conf['classifier_depth']
    conf['is_3d'] = False
    conf['interval_log'] = 64
    conf['interval_preview'] = 512

    conf['class_count'] = 10
    conf['color_channels'] = 3 if dataset_name == "cifar" else 1
    if conf['epochs'] == 0:
        conf['epochs'] = 10
    conf['omega'] = 10.0

    param_groups_siren = []
    param_groups_cls = []
    is_3d = conf['is_3d']

    # assert this is actual image size
    # CH, CW = dataset_train[0][0].shape[1:] # [C, H, W]
    # assert (H, W) == (CH, CW), f'expected {H}x{W} but got {CH}x{CW}'
    img_size = 32 if dataset_name == "cifar" else 28
    conf['sample_size'] = int(img_size*img_size * conf['subsample_points'])

    model_path = os.path.join(base_path, "basemodels", "MWT", dataset_name, f"{dataset_name}_MWT.pt")
    siren_rgb = SirenModel(ch_in=2, 
                            ch_hiddens=conf['siren_dim'], 
                            ch_out=conf['color_channels'], 
                            omega=conf['omega']).to(device)
    param_groups_siren += [ {'params': list[Parameter](siren_rgb.parameters()), 'lr': conf['lr_siren_rgb']} ]
    PI = siren_rgb.pass_indices.shape[0]
    meta_sgd = MetaSGD(conf, PI).to(device)
    param_groups_siren += [ {'params': list(meta_sgd.parameters()), 'lr': conf['lr_sgd_lrs']} ]
    classifier = build_classifier(conf, device)
    param_groups_cls += [ {'params': list(classifier.parameters()), 'lr': conf['lr_classifier']} ]
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    
    siren_rgb.load_state_dict(state['siren_rgb'])
    classifier.load_state_dict(state['classifier'])
    meta_sgd.load_state_dict(state['meta_sgd'])

    siren_rgb.eval()
    meta_sgd.eval()
    classifier.eval()   

    # data
    _, testloader = load_data(dataset_name, base_path=os.path.join(base_path, "raw"))

    model = MWT_Pipeline(conf, siren_rgb, classifier, meta_sgd, device, PI)
    model.eval()
    model.to(device)

    # evaluation
    clean_correct, adv_correct, total = 0, 0, 0

    if kwargs.get("eps") is not None:
        eps = kwargs["eps"]
    else:
        eps = 0.3 if dataset_name == "mnist" else 0.03
    steps = kwargs.get("steps", 20)
    norm = kwargs.get("norm", "linf")
    eps_scaled = eps * 2
    gamma = 2 * eps_scaled / steps
    PGD_MAP = {"linf": PGD,
               "l2": PGDL2,
               "l1": PGDL1}

    attack = PGD_MAP[norm](model=model, 
                eps=eps_scaled, 
                alpha=gamma,
                steps=steps,
                img_scale=[-1, 1]
                )


    print(f"PGD attack initialized with eps={eps}, alpha={gamma}, steps={steps}, norm={norm}")
    loss_fn = nn.CrossEntropyLoss()
    # builder = BatchedINRBuilder(batch_size=64, sidelen=32 if dataset_name == "cifar" else 28, hidden_features=32, hidden_layers=1, c_out=3 if dataset_name == "cifar" else 1, steps=5000, lr=5e-4, amp=True, device=device)
    for i, (x, y) in tqdm(enumerate(testloader)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        b = x.shape[0]

        with torch.enable_grad():
            logits_clean = model(x)

            pred_clean = logits_clean.argmax(dim=1)
            clean_correct += (pred_clean == y).sum().item()
            x_adv = attack(x, y)

            pred_eps = infer_eps_linf(x, x_adv, input_scale='-1,1')[f'eps_{norm}_max_pixels']
            print(f"PGD eps_{norm}: {pred_eps}")

            logits_adv = model(x_adv)
            pred_adv = logits_adv.argmax(dim=1)
            adv_correct += (pred_adv == y).sum().item()
        img_side_by_side(x, x_adv)
        total += y.numel()
        print(f"Batch {i} | Clean acc: {clean_correct / max(total,1):.4f} | Adv acc: {adv_correct / max(total,1):.4f}")
    print(f"Hypernet surrogate — Dataset= {dataset_name} | Clean acc: {clean_correct / max(total,1):.4f} | Adv acc: {adv_correct / max(total,1):.4f}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="mnist")
    args = args.parse_args()
    main(args.dataset_name)


