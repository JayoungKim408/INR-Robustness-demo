"""Precompute the attack for all 60 candidate MNIST test images x several eps x both victims (-> web/data/sweep.json).

    python demo/run_demo.py                       # all eps in DEFAULT_EPS, merged into web/data/sweep.json
    python demo/run_demo.py --eps 0.1 0.3         # only these eps (existing entries are kept)

Per image and victim:  clean prediction on the stored SIREN  vs  prediction on a SIREN refitted to the PGD-perturbed image.
The static page (web/index.html) reads this file, so it works on GitHub Pages without a backend.
"""
import argparse
import base64
import json
import os
import time

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import ASSETS, DEVICE, RESTARTS, STEPS, VICTIMS, Engine, stack_wsfeat

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
DEFAULT_EPS = [0.05, 0.1, 0.15, 0.2, 0.3]


def b64(t01):
    """(N,28,28) in [0,1] -> list of base64 strings of raw uint8 bytes."""
    a = (t01.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return [base64.b64encode(x.tobytes()).decode() for x in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="+", default=DEFAULT_EPS)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", 16)))

    samples = torch.load(f"{ASSETS}/samples.pt", weights_only=False)
    eng = Engine(samples)
    x = torch.stack([s["image"] for s in samples]).to(DEVICE)
    y = torch.tensor([s["label"] for s in samples]).to(DEVICE)
    clean_inr = stack_wsfeat(samples).to(DEVICE)

    path = os.path.join(WEB, "data", "sweep.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.update(steps=STEPS, restarts=RESTARTS, norm="Linf")
    out["images"] = [dict(idx=s["global_idx"], label=s["label"], img=b64(((s["image"] + 1) / 2))[0]) for s in samples]
    out.setdefault("victims", {})
    for name, title in VICTIMS.items():
        v = out["victims"].setdefault(name, dict(title=title, eps={}))
        for eps in args.eps:
            t0 = time.time()
            r = eng.evaluate(name, x, y, eps, clean_inr)
            rows = []
            for j in range(len(samples)):
                rows.append(dict(
                    probs=[round(float(p), 4) for p in r["probs_adv"][j]],
                    adv=b64((r["x_adv"][j:j + 1, 0] + 1) / 2)[0], rec=b64(r["rec_adv"][j:j + 1])[0]))
            v["eps"][str(eps)] = rows
            print(f"[{name}] eps={eps}: {len(samples)} images in {time.time() - t0:.0f}s", flush=True)
        v["clean"] = [dict(probs=[round(float(p), 4) for p in r["probs_clean"][j]], rec=b64(r["rec_clean"][j:j + 1])[0])
                      for j in range(len(samples))]
    out["eps_list"] = sorted({float(e) for vv in out["victims"].values() for e in vv["eps"]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print("wrote", path, round(os.path.getsize(path) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
