# INR-Robustness-demo

Small demo for [INR-Robustness](https://github.com/Trustworthy-and-Responsible-AI-Lab/INR-Robustness) (ICML 2026): attacking INR-based
classifiers (**DWS**, **ScaleGMN**) on MNIST with a hypernetwork surrogate. Three test images are shown with the true class,
the prediction on the original image, and the prediction on the adversarial image.

- View precomputed results: `cd demo/web && python -m http.server 8000` (or serve `demo/web` with GitHub Pages)
- Re-run the attack: `conda env create -f environment.yml && conda activate inr && python demo/run_demo.py --ids 60012 60016 60031`

See [demo/README.md](demo/README.md) for details. `src/` is a trimmed copy of the project code needed to run the demo.
