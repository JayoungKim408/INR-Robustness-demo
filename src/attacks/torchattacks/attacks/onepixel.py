import numpy as np

import torch
import torch.nn.functional as F

from ..attack import Attack
from ._differential_evolution import differential_evolution, DifferentialEvolutionSolver
from attacks.utils import SimpleTimer

class OnePixel(Attack):
    r"""
    Attack in the paper 'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]

    Modified from "https://github.com/DebangLi/one-pixel-attack-pytorch/" and 
    "https://github.com/sarathknv/adversarial-examples-pytorch/blob/master/one_pixel_attack/"

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        pixels (int): number of pixels to change (Default: 1)
        steps (int): number of steps. (Default: 10)
        popsize (int): population size, i.e. the number of candidate agents or "parents" in differential evolution (Default: 10)
        inf_batch (int): maximum batch size during inference (Default: 128)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, 
                 model, 
                 pixels=1, 
                 steps=10, 
                 popsize=10, 
                 inf_batch=128
                 ):
        super().__init__("OnePixel", model)
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, model=None):
        r"""
        Overridden.
        """
        if model is not None: # lazy set model
            self.set_model(model)
            self.set_device(next(model.parameters()).device)


        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        batch_size, channel, height, width = images.shape

        bounds = [(0, height), (0, width)] + [(0, 1)] * channel
        bounds = bounds * self.pixels

        popmul = max(1, int(self.popsize / len(bounds)))
        timer = SimpleTimer(total_steps=batch_size)
        adv_images = []
        for idx in range(batch_size):
            image, label = images[idx : idx + 1], labels[idx : idx + 1]

            if self.targeted:
                target_label = target_labels[idx : idx + 1]
                def func(delta):
                    return self._loss(image, target_label, delta)
                def callback(delta, convergence):
                    return self._attack_success(image, target_label, delta)
            else:
                def func(delta):
                    return self._loss(image, label, delta)
                def callback(delta, convergence):
                    return self._attack_success(image, label, delta)
                
            delta = differential_evolution(
                func=func,
                bounds=bounds,
                callback=callback,
                maxiter=self.steps,
                popsize=popmul,
                init="random",
                recombination=1,
                atol=-1,
                polish=False,
            ).x
            delta = np.split(delta, len(delta) / len(bounds))
            adv_image = self._perturb(image, delta)
            adv_images.append(adv_image)

            # if idx % 10 == 0 and idx > 0:
            # print(f"Batch {idx + 1}/{batch_size} completed. ETA: {timer.get_eta(idx + 1)}s")

        adv_images = torch.cat(adv_images)
        return adv_images

    def _loss(self, image, label, delta):
        adv_images = self._perturb(image, delta)  # Mutiple delta
        prob = self._get_prob(adv_images)[:, label]
        if self.targeted:
            return 1 - prob  # If targeted, increase prob
        else:
            return prob  # If non-targeted, decrease prob

    def _attack_success(self, image, label, delta):
        adv_image = self._perturb(image, delta)  # Single delta
        prob = self._get_prob(adv_image)
        pre = np.argmax(prob)
        if self.targeted and (pre == label):
            return True
        elif (not self.targeted) and (pre != label):
            return True
        return False

    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.get_logits(batch)
                outs.append(out)
        outs = torch.cat(outs)
        prob = F.softmax(outs, dim=1)
        return prob.detach().cpu().numpy()

    def _perturb(self, image, delta):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image] * num_delta, dim=0)
        for idx in range(num_delta):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                pos_x, pos_y = pixel[:2]
                channel_v = pixel[2:]
                for channel, v in enumerate(channel_v):
                    adv_images[idx, channel, int(pos_x), int(pos_y)] = v
        return adv_images
    





class BatchedOnePixelRunner:
    """
    여러 샘플의 One-Pixel DE를 동시에 실행하되,
    각 세대의 forward를 모아서 한번에 처리
    """
    def __init__(self, pipeline, pixels=1, steps=75, popsize=400):
        self.pipeline = pipeline
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize

    def attack_batch(self, images, labels, init_params):
        import time
        
        N, C, H, W = images.shape
        bounds = [(0, H), (0, W)] + [(-1, 1)] * C
        bounds = bounds * self.pixels
        n_bounds = len(bounds)
        popmul = max(1, self.popsize // n_bounds)
        pop_size = popmul * n_bounds

        from attacks.torchattacks.attacks._differential_evolution import DifferentialEvolutionSolver
        
        solvers = []
        for i in range(N):
            solver = DifferentialEvolutionSolver(
                func=None,
                bounds=bounds,
                maxiter=self.steps,
                popsize=popmul,
                init="random",
                recombination=1,
                atol=-1,
                polish=False,
                seed=None,
            )
            solvers.append(solver)

        t_dither, t_mutate, t_perturb, t_expand, t_forward, t_select = 0, 0, 0, 0, 0, 0

        for step in range(self.steps + 1):
            # --- Dithering ---
            t0 = time.time()
            for solver in solvers:
                if solver.dither is not None:
                    solver.scale = (
                        solver.random_number_generator.rand() 
                        * (solver.dither[1] - solver.dither[0]) 
                        + solver.dither[0]
                    )
            t1 = time.time()
            t_dither += (t1 - t0)

            if step % 10 == 0:
                for dbg_i, dbg_solver in enumerate(solvers[:2]):
                    print(f"  Step {step}, Sample {dbg_i}: "
                        f"best_energy={dbg_solver.population_energies[0]:.4f}, "
                        f"worst_energy={dbg_solver.population_energies[-1]:.4f}")

            # --- Mutation/Crossover ---
            t2 = time.time()
            all_images = []
            all_indices = []
            weights_list, biases_list = init_params
            
            for si, solver in enumerate(solvers):
                if step == 0:
                    candidates = solver.population
                else:
                    candidates = []
                    for c in range(solver.num_population_members):
                        trial = solver._mutate(c)
                        solver._ensure_constraint(trial)
                        candidates.append(trial)
                    candidates = np.array(candidates)
                    solver._current_trials = candidates
                
                params = np.array([solver._scale_parameters(c) for c in candidates])
                t3_inner = time.time()
                perturbed = self._perturb_batch(images[si:si+1], params)
                t4_inner = time.time()
                t_perturb += (t4_inner - t3_inner)
                
                all_images.append(perturbed)
                all_indices.append((si, perturbed.shape[0]))
            
            t3 = time.time()
            t_mutate += (t3 - t2 - (t_perturb if step == 0 else (t4_inner - t3_inner) * N))

            # --- Expand init_params ---
            t4 = time.time()
            total_size = sum(count for _, count in all_indices)
            exp_weights = []
            exp_biases = []
            for wi in range(len(weights_list)):
                parts = []
                for si, count in all_indices:
                    parts.append(weights_list[wi][si:si+1].expand(count, *weights_list[wi].shape[1:]))
                exp_weights.append(torch.cat(parts, dim=0))
            for bi in range(len(biases_list)):
                parts = []
                for si, count in all_indices:
                    parts.append(biases_list[bi][si:si+1].expand(count, *biases_list[bi].shape[1:]))
                exp_biases.append(torch.cat(parts, dim=0))
            
            self.pipeline.init_params = (tuple(exp_weights), tuple(exp_biases))
            all_images_cat = torch.cat(all_images, dim=0)
            t5 = time.time()
            t_expand += (t5 - t4)

            # --- Forward ---
            t6 = time.time()
            with torch.no_grad():
                logits = self.pipeline(all_images_cat)
                probs = torch.softmax(logits, dim=1)
            torch.cuda.synchronize()
            t7 = time.time()
            t_forward += (t7 - t6)

            # --- Selection ---
            t8 = time.time()
            offset = 0
            for si, (solver_idx, count) in enumerate(all_indices):
                current_solver = solvers[solver_idx]
                sample_probs = probs[offset:offset+count]
                label_probs = sample_probs[:, labels[si]].cpu().numpy()
                
                if step == 0:
                    current_solver.population_energies = label_probs
                    minval = np.argmin(current_solver.population_energies)
                    current_solver.population_energies[[0, minval]] = current_solver.population_energies[[minval, 0]]
                    current_solver.population[[0, minval]] = current_solver.population[[minval, 0]]
                else:
                    trials = current_solver._current_trials
                    for c in range(count):
                        if label_probs[c] < current_solver.population_energies[c]:
                            current_solver.population[c] = trials[c]
                            current_solver.population_energies[c] = label_probs[c]
                            if label_probs[c] < current_solver.population_energies[0]:
                                current_solver.population_energies[0] = label_probs[c]
                                current_solver.population[0] = trials[c]
                
                offset += count
            t9 = time.time()
            t_select += (t9 - t8)

        # --- Profiling summary ---
        total = t_dither + t_mutate + t_perturb + t_expand + t_forward + t_select
        print(f"\n[PROFILE] Total: {total:.1f}s | "
            f"dither: {t_dither:.1f}s ({t_dither/total*100:.1f}%) | "
            f"mutate: {t_mutate:.1f}s ({t_mutate/total*100:.1f}%) | "
            f"perturb: {t_perturb:.1f}s ({t_perturb/total*100:.1f}%) | "
            f"expand: {t_expand:.1f}s ({t_expand/total*100:.1f}%) | "
            f"forward: {t_forward:.1f}s ({t_forward/total*100:.1f}%) | "
            f"select: {t_select:.1f}s ({t_select/total*100:.1f}%)\n")

        # --- 최종 결과 수집 ---
        adv_images = []
        for si, solver in enumerate(solvers):
            best_delta = solver.x
            delta_split = np.split(best_delta, len(best_delta) / n_bounds)
            adv_img = self._perturb_single(images[si:si+1], delta_split)
            adv_images.append(adv_img)
        
        return torch.cat(adv_images, dim=0)

        
    def _perturb_batch(self, image, deltas):
        deltas = np.array(deltas)
        if len(deltas.shape) < 2:
            deltas = np.array([deltas])
        num = len(deltas)
        adv = image.expand(num, -1, -1, -1).clone()
        
        pixel_info = deltas.reshape(num, self.pixels, -1)
        H, W = image.shape[2], image.shape[3]
        for p in range(self.pixels):
            xs = np.clip(pixel_info[:, p, 0].astype(int), 0, H - 1)
            ys = np.clip(pixel_info[:, p, 1].astype(int), 0, W - 1)
            for c in range(image.shape[1]):
                vals = torch.from_numpy(
                    pixel_info[:, p, 2 + c].astype(np.float32)
                ).to(image.device)
                adv[range(num), c, xs, ys] = vals
        
        return adv

    # def _perturb_single(self, image, delta_split):
    #     adv = image.clone()
    #     for delta in delta_split:
    #         pixel_info = delta.reshape(self.pixels, -1)
    #         for pixel in pixel_info:
    #             x, y = int(pixel[0]), int(pixel[1])
    #             for c, v in enumerate(pixel[2:]):
    #                 adv[0, c, x, y] = v
    #     return adv
    def _perturb_single(self, image, delta_split):
        adv = image.clone()
        for delta in delta_split:
            pixel_info = delta.reshape(self.pixels, -1)
            for pixel in pixel_info:
                x = min(int(pixel[0]), image.shape[2] - 1)
                y = min(int(pixel[1]), image.shape[3] - 1)
                for c, v in enumerate(pixel[2:]):
                    adv[0, c, x, y] = v
        return adv
