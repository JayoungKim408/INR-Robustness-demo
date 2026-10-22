# INR-Robustness-demo

Small demo for [INR-Robustness](https://github.com/Trustworthy-and-Responsible-AI-Lab/INR-Robustness) (ICML 2026): attacking INR-based
classifiers (**DWS**, **ScaleGMN**) on MNIST with a hypernetwork surrogate. Three test images are shown with the true class,
the prediction on the original image, and the prediction on the adversarial image.

- Explore precomputed attacks: `cd demo/web && python -m http.server 8000`
- Re-run the attack: `conda env create -f environment.yml && conda activate inr && python demo/serve.py`

See [demo/README.md](demo/README.md) for details. `src/` is a trimmed copy of the project code needed to run the demo.
