"""
AutoAttack for INR weight space.
 
Adapts the AutoAttack framework (Croce & Hein, ICML 2020) to INR-based classifiers.
Runs three attacks sequentially in INR weight space under ICOP projection:
  1. APGD-CE  (gradient-based, cross-entropy loss)
  2. APGD-DLR (gradient-based, DLR loss)
  3. Square Attack (gradient-free, random search)
 
FAB is excluded as its minimum-norm formulation requires jointly optimizing
perturbation magnitude across weight and image spaces.
 
All attacks share the same image-space Linf constraint enforced via ICOP.
Samples fooled by earlier attacks are excluded from later attacks.
"""
 
import time
import torch
import torch.nn as nn
from tqdm import tqdm
 
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
 
# These imports assume the files are in the same directory or on PYTHONPATH
from attacks.torchattacks.autoattack.apgd import APGDAttackINR
from attacks.torchattacks.autoattack.inr_square_attack import SquareAttackINR
 
 
def _slice_wsfeat(wsfeat, indices):
    """Slice WeightSpaceFeatures by index tensor."""
    weights = tuple(w[indices] for w in wsfeat.weights)
    biases = tuple(b[indices] for b in wsfeat.biases)
    return WeightSpaceFeatures(weights, biases)
 
 
class AutoAttackINR:
    """
    AutoAttack ensemble for INR weight space.
 
    Runs APGD-CE, APGD-DLR, and Square Attack sequentially.
    Each attack only targets samples not yet fooled by previous attacks.
 
    Args:
        model:          classifier (victim_wrapper) that takes WeightSpaceFeatures
        renderer:       function that renders INR params to images
        eps:            image-space Linf bound (already scaled for data range)
        n_iter:         iterations per APGD restart (default: 100)
        n_restarts:     restarts for APGD (default: 5)
        n_queries:      queries for Square Attack (default: 100)
        attacks_to_run: list of attacks to run (default: all three)
        seed:           random seed
        verbose:        print progress
        device:         torch device
    """
 
    def __init__(self, model, renderer, eps,
                 n_iter=100, n_restarts=5, n_queries=100,
                 attacks_to_run=None, seed=0, verbose=True, device='cuda'):
        self.model = model
        self.renderer = renderer
        self.eps = eps
        self.verbose = verbose
        self.device = device
 
        if attacks_to_run is None:
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'square']
        else:
            self.attacks_to_run = attacks_to_run
 
        # Initialize individual attacks
        self.apgd_ce = APGDAttackINR(
            model, renderer, eps=eps, n_iter=n_iter, n_restarts=n_restarts,
            loss='ce', seed=seed, verbose=verbose, device=device
        )
        self.apgd_dlr = APGDAttackINR(
            model, renderer, eps=eps, n_iter=n_iter, n_restarts=n_restarts,
            loss='dlr', seed=seed, verbose=verbose, device=device
        )
        self.square = SquareAttackINR(
            model, renderer, eps=eps, n_queries=n_queries,
            seed=seed, verbose=verbose, device=device
        )
 
    def run_evaluation(self, wsfeat, labels):
        """
        Run full AutoAttack evaluation on a batch of INR weights.
 
        Args:
            wsfeat:  WeightSpaceFeatures for the full batch
            labels:  [N] ground truth labels
 
        Returns:
            dict with robust accuracy after each attack and overall
        """
        labels = labels.detach().clone().long().to(self.device)
        N = labels.shape[0]
 
        # Initial clean accuracy
        with torch.no_grad():
            logits = self.model(wsfeat)
            preds = logits.max(1)[1]
            robust_flags = (preds == labels)
 
        clean_acc = robust_flags.float().mean().item()
        results = {'clean_acc': clean_acc}
 
        if self.verbose:
            print(f'[AutoAttack-INR] clean accuracy: {clean_acc:.2%} '
                  f'({robust_flags.sum().item()}/{N})')
 
        startt = time.time()
 
        # Store adversarial WeightSpaceFeatures
        # (start with clean, update as attacks succeed)
        adv_weights = [w.clone() for w in wsfeat.weights]
        adv_biases = [b.clone() for b in wsfeat.biases]
 
        for attack_name in self.attacks_to_run:
            num_robust = robust_flags.sum().item()
            if num_robust == 0:
                if self.verbose:
                    print(f'[{attack_name}] no robust samples remaining, skipping')
                results[attack_name] = 0.0
                continue
 
            # Subset to still-robust samples
            robust_idx = robust_flags.nonzero().squeeze()
            if robust_idx.dim() == 0:
                robust_idx = robust_idx.unsqueeze(0)
 
            wsfeat_sub = _slice_wsfeat(wsfeat, robust_idx)
            labels_sub = labels[robust_idx]
 
            if self.verbose:
                print(f'\n[AutoAttack-INR] running {attack_name} on '
                      f'{num_robust} samples...')
 
            # Select and run attack
            if attack_name == 'apgd-ce':
                adv_wsfeat_sub = self.apgd_ce.perturb(wsfeat_sub, labels_sub)
            elif attack_name == 'apgd-dlr':
                adv_wsfeat_sub = self.apgd_dlr.perturb(wsfeat_sub, labels_sub)
            elif attack_name == 'square':
                adv_wsfeat_sub = self.square.perturb(wsfeat_sub, labels_sub)
            else:
                raise ValueError(f'Unknown attack: {attack_name}')
 
            # Check which samples were fooled
            with torch.no_grad():
                adv_logits = self.model(adv_wsfeat_sub)
                adv_preds = adv_logits.max(1)[1]
                fooled = (adv_preds != labels_sub)
 
            # Update robust flags and adversarial samples
            fooled_global_idx = robust_idx[fooled]
            robust_flags[fooled_global_idx] = False
 
            # Update adversarial weights for fooled samples
            for layer_i in range(len(adv_weights)):
                adv_weights[layer_i][fooled_global_idx] = \
                    adv_wsfeat_sub.weights[layer_i][fooled].detach().clone()
                adv_biases[layer_i][fooled_global_idx] = \
                    adv_wsfeat_sub.biases[layer_i][fooled].detach().clone()
 
            robust_acc = robust_flags.float().mean().item()
            results[attack_name] = robust_acc
 
            if self.verbose:
                print(f'[AutoAttack-INR] robust accuracy after {attack_name}: '
                      f'{robust_acc:.2%} ({robust_flags.sum().item()}/{N}) - '
                      f'time: {time.time() - startt:.1f}s')
 
        results['final_robust_acc'] = robust_flags.float().mean().item()
        if self.verbose:
            print(f'\n[AutoAttack-INR] final robust accuracy: '
                  f'{results["final_robust_acc"]:.2%} - '
                  f'total time: {time.time() - startt:.1f}s')
 
        return results
 