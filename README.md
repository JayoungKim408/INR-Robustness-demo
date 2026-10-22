# INR-Robustness-demo

Interactive demo for [INR-Robustness](https://github.com/Trustworthy-and-Responsible-AI-Lab/INR-Robustness) (ICML 2026):
attacking INR-based classifiers on MNIST with a **hypernetwork surrogate**.

The victims (**DWS**, **ScaleGMN**) don't see pixels. They classify a SIREN (an INR) fitted to the image. The attacker only edits the image:

1. A hypernetwork maps an image to SIREN weights, so `image -> hypernetwork -> SIREN weights -> victim` is differentiable.
2. L-inf PGD (20 steps, best of 5 restarts) on that chain produces an adversarial image.
3. A fresh SIREN (5000 Adam steps, same init) is fitted to the adversarial image and the real victim classifies it.

The page compares, for each image and victim, the true class, the prediction on the original image, and the prediction on the adversarial image.

## What you can do

| | Needs | Where |
|---|---|---|
| **Explore** 60 MNIST test images, slide epsilon (0.05 - 0.3), see original / adversarial image, perturbation, class probabilities and success rates | a static web server | locally or GitHub Pages |
| **Draw your own digit** and attack it live | the Python backend (`serve.py`) | locally |

## Quick start

### 0. Environment
Conda env `inr` (Python 3.11, PyTorch 2.5.1 + CUDA 12.4, torch_geometric):
```bash
conda env create -f environment.yml     # once; creates the env "inr"
conda activate inr
```

### 1. Explore only (no GPU, no Python deps)
```bash
cd demo/web && python -m http.server 8000
```
Open <http://localhost:8000>. This reads the precomputed `demo/web/data/sweep.json`.

### 2. Explore + draw your own digit (live backend)
```bash
conda activate inr
python demo/serve.py                    # uses a GPU if available
CUDA_VISIBLE_DEVICES=2 python demo/serve.py    # pick a specific GPU
CUDA_VISIBLE_DEVICES= python demo/serve.py     # force CPU
python demo/serve.py --port 8123        # if 8000 is taken
```
Open <http://127.0.0.1:8000>, draw a digit and press **Attack**.

- One request fits 3 SIRENs x 5000 steps: about **a minute on a GPU**, several minutes on CPU.
- A drawn digit has no label, so each victim is attacked away from its own clean prediction.
- Requests are processed one at a time.
- The server binds to `127.0.0.1` only.

**On a remote machine**, forward the port from your local computer, keep that SSH session open, and start `serve.py` inside it:
```bash
ssh -L 8000:localhost:8000 <user>@<server>     # local terminal
# in that session:
conda activate inr && cd INR-Robustness-demo && python demo/serve.py
# then open http://127.0.0.1:8000 in your local browser
```
Use the same port number on both sides of `-L` if you change `--port`.

### 3. GitHub Pages
Pages serves only the repo root or `/docs`, so copy the web folder and enable Pages (Settings -> Pages -> branch, folder `/docs`):
```bash
cp -r demo/web docs && git add docs && git commit -m "Add docs for Pages" && git push
```
On Pages the "Draw your own" panel shows a note instead, since there is no backend.

## Regenerate the precomputed results
```bash
python demo/run_demo.py                 # all epsilons, ~6 min on one GPU (CPU works, slower)
python demo/run_demo.py --eps 0.1 0.3   # only these; existing entries are kept
```
Rewrites `demo/web/data/sweep.json`. If you changed the web folder copy in `docs/`, copy it again.

## Layout
```
demo/
  web/index.html, web/data/sweep.json   the page and its precomputed data
  engine.py                             model loading + the attack (shared by the two scripts below)
  run_demo.py                           precompute sweep.json
  serve.py                              live backend
  assets/ckpt/                          DWS and ScaleGMN classifiers + their hypernetwork surrogates (~126 MB)
  assets/samples.pt                     60 MNIST test images with their fitted SIRENs
  make_assets.py                        rebuild assets/ from the full project tree (needs the original data)
src/                                    trimmed copy of the INR-Robustness code needed to run the demo
environment.yml
```

## Results and caveats
Attack success is **far from guaranteed**. On the 60 images here at epsilon 0.3 it fools DWS on about 73% and ScaleGMN on about 14% of
the correctly-classified images (the page shows these numbers per epsilon). Over the full MNIST test set, the project's runs give
accuracy under this attack of 30.2% (DWS) and 78.8% (ScaleGMN). DWS is also only about 88% accurate on clean images, so its
misclassified-clean cases are excluded from the success counts.

Outcomes of a single image can vary slightly between runs (random PGD start, GPU nondeterminism).

## Citation
Please cite the INR-Robustness paper (ICML 2026) when using this demo; see the main repository for details.
