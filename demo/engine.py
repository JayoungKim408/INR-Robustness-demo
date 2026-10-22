"""Shared model loading and the hypernetwork-surrogate attack, used by run_demo.py (precompute) and serve.py (live)."""
import os
import sys

import torch

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # CPU works too; CUDA_VISIBLE_DEVICES= forces it
STEPS = 20
RESTARTS = 5            # project default for the surrogate attacks
PREFIX = "fixed_init_bz_no_isfirst"
VICTIMS = {"dws": "DWS", "scalegmn": "ScaleGMN"}

# BatchedINRBuilder calls torch.cuda.synchronize() unconditionally; make it a no-op without CUDA.
if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None


def stack_wsfeat(samples, prefix=""):
    """list of per-sample (weights, biases) -> batched WeightSpaceFeatures (same layout as the dataset)."""
    w = tuple(torch.stack([s[prefix + "weights"][i] for s in samples]) for i in range(len(samples[0][prefix + "weights"])))
    b = tuple(torch.stack([s[prefix + "biases"][i] for s in samples]) for i in range(len(samples[0][prefix + "biases"])))
    return WeightSpaceFeatures(w, b)


class Engine:
    def __init__(self, samples):
        self.samples = samples
        self.victims, self.surrogates = {}, {}
        cfg = INRConfig(depth=1, width=32, scale=30, extra_aug=True, prefix=PREFIX)
        ws, init = stack_wsfeat(samples[:2]), stack_wsfeat(samples[:2], "init_")
        batch = [((ws.weights, ws.biases), (init.weights, init.biases), None, None)]  # only used to infer the network spec
        for name in VICTIMS:
            v = load_victim_model(
                victim_model=name, dataset_name="mnist", path_config=PathConfig(base_path=ROOT), inr_config=cfg,
                device=DEVICE, ckpt_path={"basemodel": "-"}, testloader=batch, load_state=False,
                return_wrapped=True, attack_space="image")
            v.model.load_state_dict(torch.load(f"{CKPT}/{name}.pt", map_location="cpu"))
            for p in v.parameters():
                p.requires_grad_(False)
            v.eval()
            hyper = ConvolutionalNeuralProcessImplicit2DHypernet(
                in_features=1, out_features=1, image_resolution=(28, 28), hidden_features=32, num_hidden_layers=1)
            hyper.load_state_dict(torch.load(f"{CKPT}/{name}_hypernet.pth", map_location="cpu"))
            for p in hyper.parameters():
                p.requires_grad_(False)
            self.victims[name] = v
            self.surrogates[name] = SurrogateClassifier(hyper.eval(), v).eval().to(DEVICE)
        # every sample shares the same SIREN init, so any image can be fitted from it
        self.init = stack_wsfeat(samples[:1], "init_").to(DEVICE)
        self.builder = BatchedINRBuilder(sidelen=28, hidden_features=32, hidden_layers=1, first_omega_0=30.,
                                         hidden_omega_0=30., c_out=1, steps=5000, lr=5e-4, amp=False, device=DEVICE)

    def fit(self, x):
        """Fit one SIREN per image (x in [-1,1], shape B,1,28,28) from the shared init."""
        init = WeightSpaceFeatures(tuple(w.expand(len(x), *w.shape[1:]) for w in self.init.weights),
                                   tuple(b.expand(len(x), *b.shape[1:]) for b in self.init.biases))
        return WeightSpaceFeatures(*self.builder.build_wts_and_bs_from_img(x, init))

    def attack(self, name, x, y, eps, seed=0):
        """Multi-restart Linf PGD on the image through hypernet -> SIREN -> victim. eps in [0,1] pixel scale."""
        torch.manual_seed(seed)
        sur = self.surrogates[name]
        e = eps * 2                                                  # images live in [-1, 1]
        pgd = PGD(model=sur, eps=e, alpha=2 * e / STEPS, steps=STEPS, img_scale=[-1, 1])
        x_adv, best = x.clone(), torch.full((len(x),), -float("inf"), device=x.device)
        for _ in range(RESTARTS):
            with torch.enable_grad():
                cand = pgd(x, y)
            with torch.no_grad():
                loss = torch.nn.functional.cross_entropy(sur(cand), y, reduction="none")
            upd = loss > best
            x_adv[upd], best[upd] = cand[upd], loss[upd]
        return x_adv

    def evaluate(self, name, x, y, eps, clean_inr, seed=0):
        """Returns cpu tensors: x_adv, probs_clean, probs_adv, rec_clean, rec_adv (renders in [0,1])."""
        v = self.victims[name]
        x_adv = self.attack(name, x, y, eps, seed)
        adv_inr = self.fit(x_adv)
        with torch.no_grad():
            pc, pa = v(clean_inr).softmax(1), v(adv_inr).softmax(1)
            rc = self.builder.build_img_from_wts_and_bs(clean_inr, if_wsfeat=True).squeeze(-1)
            ra = self.builder.build_img_from_wts_and_bs(adv_inr, if_wsfeat=True).squeeze(-1)
        cpu = lambda t: t.detach().cpu()
        return dict(x_adv=cpu(x_adv), probs_clean=cpu(pc), probs_adv=cpu(pa), rec_clean=cpu(rc), rec_adv=cpu(ra))
