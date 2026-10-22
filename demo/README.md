# Demo: hypernetwork-surrogate attack on INR classifiers (MNIST)

Attacks **DWS** and **ScaleGMN** (classifiers that read a SIREN fitted to an image) through a hypernetwork surrogate
(`attack_type: i2i-hypernet`, image-space L-inf PGD, 20 steps, best of 5 restarts):

1. PGD on the image through `image -> hypernetwork -> SIREN weights -> victim` (differentiable surrogate).
2. Fit a fresh SIREN (5000 Adam steps, same init) to the adversarial image.
3. The real victim classifies that SIREN.

## 1. Explore (static, works on GitHub Pages)
Pick any of 60 MNIST test images and an epsilon in {0.05, 0.1, 0.15, 0.2, 0.3}; see original / adversarial image,
perturbation, class probabilities and per-epsilon success rates for both victims. Reads precomputed `web/data/sweep.json`.
```bash
cd demo/web && python -m http.server 8000      # http://localhost:8000
```
For GitHub Pages, serve `demo/web` (e.g. copy it to `docs/`).

## 2. Draw your own digit (live, needs the backend)
```bash
conda env create -f environment.yml && conda activate inr
python demo/serve.py                     # http://127.0.0.1:8000, uses a GPU if available
CUDA_VISIBLE_DEVICES= python demo/serve.py   # force CPU
```
Draw a digit and press Attack. One request fits 3 SIRENs x 5000 steps (about a minute on a GPU, several on CPU).
A drawn digit has no label, so each victim is attacked away from its own clean prediction.
Remote machine: `ssh -L 8000:localhost:8000 <host>`.

## Regenerate the precomputed results
```bash
python demo/run_demo.py                  # all epsilons, ~6 min on one GPU (CPU works, slower)
python demo/run_demo.py --eps 0.1 0.3    # only some; existing entries are kept
```

## Contents
- `web/index.html`, `web/data/sweep.json`: the page and its data.
- `engine.py`: model loading + attack; `run_demo.py`: precompute; `serve.py`: live server.
- `assets/ckpt/`: DWS and ScaleGMN classifiers and their hypernetwork surrogates; `assets/samples.pt`: 60 MNIST test images with fitted SIRENs.
- `make_assets.py`: rebuilds `assets/` from the full project tree (needs the original data).

## Caveat
The attack is far from always successful. On these 60 images at epsilon 0.3 it fools DWS on ~73% and ScaleGMN on ~14% of the
correctly-classified images; over the full MNIST test set the project's runs give accuracy under attack of 30.2% (DWS) and 78.8% (ScaleGMN).
DWS is only ~88% accurate on clean images.
