"""Hypernetwork-surrogate attack demo on MNIST (DWS and ScaleGMN victims). Runs on CPU or one GPU.

For each chosen MNIST test image and each victim:
  1. clean:  victim classifies the stored SIREN fitted to the original image
  2. attack: PGD (Linf) on the *image* against  hypernet -> SIREN weights -> victim  (the surrogate)
  3. adv:    fit a fresh SIREN to the perturbed image (same init), victim classifies it

    python demo/run_demo.py --ids 60001 60004 60007          # writes demo/web/results.json + images
    python demo/run_demo.py --pool 30                        # table over candidates, to pick images
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from attacks.hypernet.hypernet_attack import SurrogateClassifier  # noqa: E402
from attacks.hypernet.train_hypernet.meta_modules import ConvolutionalNeuralProcessImplicit2DHypernet  # noqa: E402
from attacks.torchattacks.attacks.pgd import PGD  # noqa: E402
from attacks.utils import BatchedINRBuilder  # noqa: E402
from basemodels.model_loading import load_victim_model  # noqa: E402
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures  # noqa: E402
from utils.inr_config import INRConfig  # noqa: E402
from utils.path_config import PathConfig  # noqa: E402

ASSETS = os.path.join(HERE, "assets")
CKPT = os.path.join(ASSETS, "ckpt")
WEB = os.path.join(HERE, "web")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # CPU works too; use CUDA_VISIBLE_DEVICES= to force it
EPS = 0.3          # Linf budget in [0,1] pixel scale (project default for MNIST)
STEPS = 20
RESTARTS = 5       # project default for the surrogate attacks
PREFIX = "fixed_init_bz_no_isfirst"
INFO = {
    "dws": dict(title="DWS"),
    "scalegmn": dict(title="ScaleGMN"),
}

# BatchedINRBuilder calls torch.cuda.synchronize() unconditionally; make it a no-op without CUDA.
if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None


def stack_wsfeat(samples, prefix=""):
    """list of per-sample (weights, biases) -> batched WeightSpaceFeatures (same layout as the dataset)."""
    w = tuple(torch.stack([s[prefix + "weights"][i] for s in samples]) for i in range(len(samples[0][prefix + "weights"])))
    b = tuple(torch.stack([s[prefix + "biases"][i] for s in samples]) for i in range(len(samples[0][prefix + "biases"])))
    return WeightSpaceFeatures(w, b)


def load_victim(name, samples):
    cfg = INRConfig(depth=1, width=32, scale=30, extra_aug=True, prefix=PREFIX)
    ws = stack_wsfeat(samples[:2])
    init = stack_wsfeat(samples[:2], "init_")
    batch = [((ws.weights, ws.biases), (init.weights, init.biases), None, None)]  # only used to infer the network spec
    victim = load_victim_model(
        victim_model=name, dataset_name="mnist", path_config=PathConfig(base_path=ROOT), inr_config=cfg,
        device=DEVICE, ckpt_path={"basemodel": "-"}, testloader=batch,
        load_state=False, return_wrapped=True, attack_space="image")
    victim.model.load_state_dict(torch.load(f"{CKPT}/{name}.pt", map_location="cpu"))
    for p in victim.parameters():
        p.requires_grad_(False)
    return victim.eval()


def load_surrogate(name, victim):
    hyper = ConvolutionalNeuralProcessImplicit2DHypernet(
        in_features=1, out_features=1, image_resolution=(28, 28), hidden_features=32, num_hidden_layers=1)
    hyper.load_state_dict(torch.load(f"{CKPT}/{name}_hypernet.pth", map_location="cpu"))
    for p in hyper.parameters():
        p.requires_grad_(False)
    return SurrogateClassifier(hyper.eval(), victim).eval()


def run_victim(name, samples, seed=0):
    torch.manual_seed(seed)
    victim = load_victim(name, samples)
    surrogate = load_surrogate(name, victim).to(DEVICE)
    x = torch.stack([s["image"] for s in samples]).to(DEVICE)       # [-1, 1]
    y = torch.tensor([s["label"] for s in samples]).to(DEVICE)
    inr_clean = stack_wsfeat(samples).to(DEVICE)
    init = stack_wsfeat(samples, "init_").to(DEVICE)

    eps = EPS * 2                                                   # images live in [-1, 1]
    attack = PGD(model=surrogate, eps=eps, alpha=2 * eps / STEPS, steps=STEPS, img_scale=[-1, 1])
    builder = BatchedINRBuilder(sidelen=28, hidden_features=32, hidden_layers=1, first_omega_0=30., hidden_omega_0=30.,
                                c_out=1, steps=5000, lr=5e-4, amp=False, device=DEVICE)

    t0 = time.time()
    with torch.no_grad():
        logit_clean = victim(inr_clean)
        sur_clean = surrogate(x)
    # multi-restart PGD: keep, per image, the restart with the highest surrogate loss
    x_adv, best = x.clone(), torch.full((len(x),), -float("inf"), device=DEVICE)
    for _ in range(RESTARTS):
        with torch.enable_grad():
            cand = attack(x, y)
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(surrogate(cand), y, reduction="none")
        upd = loss > best
        x_adv[upd], best[upd] = cand[upd], loss[upd]
    with torch.no_grad():
        sur_adv = surrogate(x_adv)
    wts, bs = builder.build_wts_and_bs_from_img(x_adv, init)
    inr_adv = WeightSpaceFeatures(wts, bs)
    with torch.no_grad():
        logit_adv = victim(inr_adv)
        rec_clean = builder.build_img_from_wts_and_bs(inr_clean, if_wsfeat=True)
        rec_adv = builder.build_img_from_wts_and_bs(inr_adv, if_wsfeat=True)
    print(f"[{name}] {len(samples)} images in {time.time() - t0:.0f}s")
    cpu = lambda t: t.detach().cpu()
    return dict(x_adv=cpu(x_adv), logit_clean=cpu(logit_clean), logit_adv=cpu(logit_adv), sur_clean=cpu(sur_clean),
                sur_adv=cpu(sur_adv), rec_clean=cpu(rec_clean.squeeze(-1)), rec_adv=cpu(rec_adv.squeeze(-1)))


def png(t01, path, scale=8):
    a = (t01.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", type=int, nargs="+", help="global dataset indices (>= 60000) to showcase")
    g.add_argument("--pool", type=int, help="evaluate the first N candidates and print a table")
    ap.add_argument("--seed", type=int, default=0, help="PGD random-start seed")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", 16)))

    all_samples = torch.load(f"{ASSETS}/samples.pt", weights_only=False)
    pools = {n: all_samples for n in INFO}
    if args.pool:
        pools = {n: s[:args.pool] for n, s in pools.items()}
    else:
        pools = {n: [s for i in args.ids for s in pool if s["global_idx"] == i] for n, pool in pools.items()}
        assert all(len(s) == len(args.ids) for s in pools.values()), "id not in candidate pool"

    res = {n: run_victim(n, pools[n], seed=args.seed) for n in INFO}

    if args.pool:
        print("idx   label | " + " | ".join(f"{n}: clean adv (sur_adv)" for n in INFO))
        for j, s in enumerate(pools["dws"]):
            row = [f"{s['global_idx']} {s['label']:>3}   "]
            for n in INFO:
                r = res[n]
                row.append(f"{r['logit_clean'][j].argmax().item()}  {r['logit_adv'][j].argmax().item()}  ({r['sur_adv'][j].argmax().item()})")
            print(" | ".join(row))
        return

    os.makedirs(f"{WEB}/img", exist_ok=True)
    out = {"eps": EPS, "steps": STEPS, "restarts": RESTARTS, "seed": args.seed, "norm": "Linf", "samples": []}
    for j, i in enumerate(args.ids):
        s0 = pools["dws"][j]
        png((s0["image"][0] + 1) / 2, f"{WEB}/img/{i}_orig.png")
        entry = {"idx": i, "label": s0["label"], "orig": f"img/{i}_orig.png", "victims": {}}
        for n in INFO:
            r = res[n]
            png((r["x_adv"][j][0] + 1) / 2, f"{WEB}/img/{i}_{n}_adv.png")
            png(((r["x_adv"][j][0] - pools[n][j]["image"][0]).abs() / 2 / (2 * EPS) ).clamp(0, 1), f"{WEB}/img/{i}_{n}_delta.png")
            for tag in ("clean", "adv"):
                png(r["rec_" + tag][j], f"{WEB}/img/{i}_{n}_rec_{tag}.png")
            pc, pa = r["logit_clean"][j].softmax(0), r["logit_adv"][j].softmax(0)
            entry["victims"][n] = {
                "title": INFO[n]["title"],
                "adv": f"img/{i}_{n}_adv.png", "delta": f"img/{i}_{n}_delta.png",
                "rec_clean": f"img/{i}_{n}_rec_clean.png", "rec_adv": f"img/{i}_{n}_rec_adv.png",
                "pred_clean": int(pc.argmax()), "conf_clean": float(pc.max()),
                "pred_adv": int(pa.argmax()), "conf_adv": float(pa.max()),
                "linf": float((r["x_adv"][j] - pools[n][j]["image"]).abs().max() / 2),
                "probs_clean": [round(float(p), 4) for p in pc], "probs_adv": [round(float(p), 4) for p in pa],
            }
        out["samples"].append(entry)
    out["footnote"] = ("These 3 images were <b>selected</b> from 60 candidates because the attack succeeds on both victims; it does not always. "
                       "Over the full 10k MNIST test set (project runs), accuracy under this attack is 30.2% for DWS and 78.8% for ScaleGMN. "
                       "DWS is also only ~88% accurate on clean fixed-init SIRENs.")
    json.dump(out, open(f"{WEB}/results.json", "w"), indent=1)
    print("wrote", f"{WEB}/results.json")


if __name__ == "__main__":
    main()
