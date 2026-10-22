"""Build the slim asset bundle for the demo from the full project directory.

Run once on a machine that has the full /mnt/data0/.../INR-Robustness tree:
    python demo/make_assets.py --base /mnt/data0/jayoung/INR-Robustness

Writes into demo/assets/:
    samples.pt                     candidate MNIST test images with their fitted SIREN weights
    ckpt/{dws,scalegmn}.pt, ckpt/{dws,scalegmn}_hypernet.pth
"""
import argparse
import os
import sys

import torch
from torchvision.datasets import MNIST

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from basemodels.utils.nfn.data_utils import SirenDataset  # noqa: E402

INR = "inr_depth1_width32_scale30"
# victim -> (siren prefix, hypernet surrogate dir)
PREFIX = "fixed_init_bz_no_isfirst"   # DWS and ScaleGMN share this INR set
VICTIMS = ["dws", "scalegmn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/data0/jayoung/INR-Robustness")
    ap.add_argument("--pool", type=int, default=60, help="number of candidate test images")
    args = ap.parse_args()

    assets = os.path.join(args.base, "assets")
    out = os.path.join(HERE, "assets")
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)

    # ---- candidate samples: first `pool` test images (global index >= 60000)
    siren_path = os.path.join(assets, "sirens", "depth1_width32_scale30", "mnist")
    mnist = MNIST(os.path.join(assets, "raw"), train=False, download=False)
    ds = SirenDataset(siren_path, siren_prefix=PREFIX, split="test", split_points=[55000, 60000])
    samples = []
    for j in range(args.pool):
        (w, b), (iw, ib), label = ds[j]
        gidx = ds.idcs[j]
        img_pil, lab = mnist[gidx - 60000]
        assert lab == label, (gidx, lab, label)
        img = torch.from_numpy(__import__("numpy").array(img_pil)).float().div(255).sub(0.5).div(0.5)[None]
        samples.append(dict(global_idx=gidx, label=label, image=img,
                            weights=[x.clone() for x in w], biases=[x.clone() for x in b],
                            init_weights=[x.clone() for x in iw], init_biases=[x.clone() for x in ib]))
    torch.save(samples, os.path.join(out, "samples.pt"))
    print("samples:", len(samples), "first idx", samples[0]["global_idx"])

    # ---- checkpoints
    bm = os.path.join(assets, "basemodels")
    hy = os.path.join(assets, "attacks", "i2i-hypernet", "mnist")
    ck = os.path.join(out, "ckpt")
    torch.save(torch.load(f"{bm}/dws/mnist/{INR}/best.pt", map_location="cpu", weights_only=False), f"{ck}/dws.pt")
    torch.save(torch.load(f"{bm}/scalegmn/mnist/{INR}/best.pt", map_location="cpu", weights_only=False),
               f"{ck}/scalegmn.pt")
    for victim in VICTIMS:  # hypernet surrogates trained against each victim
        sd = torch.load(f"{hy}/{victim}/mnist_5e-05_0.1_100_1.0/final.pth", map_location="cpu", weights_only=False)
        torch.save(sd, f"{ck}/{victim}_hypernet.pth")
    for f in sorted(os.listdir(ck)):
        print(f, round(os.path.getsize(os.path.join(ck, f)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
