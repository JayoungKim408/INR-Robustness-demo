# Demo: hypernetwork-surrogate attack on INR classifiers (MNIST)

Shows three MNIST test images attacked against **DWS** and **ScaleGMN**. For each image and victim the page compares
the true class, the prediction on the original image, and the prediction on the adversarial image.

Attack (`attack_type: i2i-hypernet`, image space, L∞ ε=0.3, 20-step PGD, 5 restarts):
1. PGD on the image through `image → hypernetwork → SIREN weights → victim` (differentiable surrogate).
2. Fit a fresh SIREN (5000 Adam steps, same init) to the adversarial image.
3. The real victim classifies that SIREN.

## View the precomputed results
```bash
cd demo/web && python -m http.server 8000     # open http://localhost:8000
```
(also works on GitHub Pages: serve `demo/web`)

## Re-run the attack (~1 min for 3 images on one GPU; CPU also works, a few minutes)
```bash
conda activate inr
python demo/run_demo.py --ids 60012 60016 60031    # any indices 60000..59 in demo/assets/samples.pt
CUDA_VISIBLE_DEVICES= python demo/run_demo.py --ids ...  # force CPU
```
This rewrites `demo/web/results.json` and `demo/web/img/`. Outcomes can shift slightly with the batch of ids you pass (random PGD start).

## Contents
- `assets/ckpt/`: DWS and ScaleGMN classifiers and their two hypernetwork surrogates (~126 MB total).
- `assets/samples.pt`: 60 MNIST test images with their fitted SIRENs (`fixed_init_bz_no_isfirst`).
- `make_assets.py`: rebuilds `assets/` from the full project tree.

## Caveat
The images shown were **selected** among 60 candidates because the attack succeeds on both victims. It is not always
successful: over the full MNIST test set, accuracy under this attack is 30.2% (DWS) and 78.8% (ScaleGMN) in the project's runs.
