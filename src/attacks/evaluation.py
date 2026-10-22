"""Evaluation of victim models on clean and adversarial datasets."""

import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from basemodels.inr2array import AutoEncoder
from basemodels.utils.nfn.models import InvariantNFN
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from attacks.utils import BatchedINRBuilder, get_dataset_params, wsfeat_to_dws
from attacks.scalegmn_utils import wsfeat_to_pyg_batch, get_scalegmn_batch_params
from defences import DiffusionDenoiser
from utils.path_config import PathConfig


def evaluate_on_perturbed_dataset(
    loader: DataLoader,
    victim: Dict[str, Any],
    victim_model_type: str,
    dataset_name: str,
    path_config: "PathConfig",
    inr_config: dict,
    device: torch.device,
    defense_type: Optional[str] = None,
    defense_params: Optional[dict] = None,
    denoiser: Optional[DiffusionDenoiser] = None,
    surrogate: Optional[nn.Module] = None,
    weight_shapes: Optional[List] = None,
    bias_shapes: Optional[List] = None,
) -> Dict[str, Any]:
    """Evaluate a victim model on clean and adversarial examples.

    Parameters
    ----------
    loader : DataLoader
        Yields ``(x, inr, embedding, y, x_adv, inr_adv)`` tuples.
    victim : dict
        Model artefacts keyed by model-specific names (e.g. ``"nfnet"``,
        ``"clf"``, ``"scalegmn"``, ``"dws"``).
    victim_model_type : str
        One of ``"nfn"``, ``"nft"``, ``"scalegmn"``, ``"mwt"``, ``"dws"``,
        ``"inr2vec"``.
    dataset_name : str
        Name of the dataset (used to look up side length / channels).
    path_config : PathConfig
        Provides ``get_inr_architecture_params``.
    inr_config : dict
        INR configuration forwarded to *path_config*.
    device : torch.device
        Compute device.
    defense_type : str or None
        Active defense type (e.g. ``"diffusion"``).
    defense_params : dict or None
        Parameters for the active defense.
    denoiser : DiffusionDenoiser or None
        Pre-built denoiser used when *defense_type* is ``"diffusion"``.
    surrogate : nn.Module or None
        Surrogate model for diffusion surrogate mode.
    weight_shapes : list or None
        Weight shapes for surrogate unflattening.
    bias_shapes : list or None
        Bias shapes for surrogate unflattening.

    Returns
    -------
    dict
        ``{"clean_acc": float, "adv_acc": float, "total": int}``
    """
    dparams = get_dataset_params(dataset_name)
    side = dparams["sidelen"]
    c_out = dparams["c_out"]
    # Get INR architecture params from config
    inr_params = path_config.get_inr_architecture_params(inr_config)
    builder = BatchedINRBuilder(
        sidelen=side,
        hidden_features=inr_params["hidden_features"],
        hidden_layers=inr_params["hidden_layers"],
        c_out=c_out,
        steps=5000,
        lr=5e-4,
        amp=True,
        device=device,
    )

    # Check defense mode
    defense_enabled = defense_type is not None
    _defense_params = defense_params if defense_params is not None else {}
    defense_mode = _defense_params.get("mode", "denoise_only") if defense_enabled else None

    # Check if we're using diffusion defense with refit mode
    diffusion_refit_mode = (
        denoiser is not None and
        defense_enabled and
        defense_type == "diffusion" and
        defense_mode == "refit"
    )

    # Check if we're using diffusion defense with surrogate mode
    diffusion_surrogate_mode = (
        denoiser is not None and
        surrogate is not None and
        defense_enabled and
        defense_type == "diffusion" and
        defense_mode == "surrogate"
    )

    if diffusion_refit_mode:
        print("  Using diffusion defense with refit mode")
        print(f"  [Eval] Refit config: steps={builder.steps}, lr={builder.lr}, amp={builder.amp}")
    elif diffusion_surrogate_mode:
        print("  Using diffusion defense with surrogate mode")

    # Timing tracking
    eval_start_time = time.time()
    total_denoise_time = 0.0
    total_refit_time = 0.0
    total_inference_time = 0.0
    batch_count = 0

    clean_correct = 0
    adv_correct = 0
    total = 0

    if victim_model_type == "nfn":
        nfnet: InvariantNFN = victim["nfnet"]
        with torch.no_grad():
            for batch in loader:
                batch_start = time.time()
                batch_count += 1
                # (x, inr, embedding, y, x_adv, inr_adv)
                x, inr, _, y, x_adv, inr_adv = batch
                x = x.to(device, non_blocking=True)
                x_adv = x_adv.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # Move nested INR tensors to device
                inr = tuple([[t.to(device) for t in j] for j in inr])
                inr_adv = tuple([[t.to(device) for t in j] for j in inr_adv])

                # Apply diffusion defense if enabled (refit mode)
                if diffusion_refit_mode:
                    denoise_start = time.time()
                    # Denoise clean image
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Refit INRs from denoised images
                    refit_start = time.time()
                    clean_wts_and_bs = builder.build_wts_and_bs_from_img(x_clean_denoised)
                    adv_wts_and_bs = builder.build_wts_and_bs_from_img(x_adv_denoised)
                    refit_time = time.time() - refit_start
                    total_refit_time += refit_time

                    inference_start = time.time()
                    clean_logits = nfnet(WeightSpaceFeatures(*clean_wts_and_bs))
                    adv_logits = nfnet(WeightSpaceFeatures(*adv_wts_and_bs))
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, refit={refit_time:.2f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")
                elif diffusion_surrogate_mode:
                    from src.attacks.surrogate.train_surrogate import unflatten_to_wsfeat

                    denoise_start = time.time()
                    # Denoise clean image
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Use surrogate to predict INR weights from denoised images
                    surrogate_start = time.time()
                    batch_size = x.shape[0]
                    # Clean: Image -> surrogate -> INR weights
                    flat_clean = x_clean_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_clean_flat = surrogate(flat_clean)
                    clean_wsfeat = unflatten_to_wsfeat(pred_clean_flat, weight_shapes, bias_shapes, device=device)

                    # Adversarial: Image -> surrogate -> INR weights
                    flat_adv = x_adv_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_adv_flat = surrogate(flat_adv)
                    adv_wsfeat = unflatten_to_wsfeat(pred_adv_flat, weight_shapes, bias_shapes, device=device)
                    surrogate_time = time.time() - surrogate_start
                    total_refit_time += surrogate_time  # Reuse refit timer for surrogate

                    inference_start = time.time()
                    clean_logits = nfnet(clean_wsfeat)
                    adv_logits = nfnet(adv_wsfeat)
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, surrogate={surrogate_time:.3f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")
                else:
                    # Use pre-computed INRs
                    clean_logits = nfnet(WeightSpaceFeatures(*inr))
                    adv_logits = nfnet(WeightSpaceFeatures(*inr_adv))

                clean_pred = clean_logits.argmax(dim=1)
                clean_correct += (clean_pred == y).sum().item()
                
                adv_pred = adv_logits.argmax(dim=1)
                adv_correct += (adv_pred == y).sum().item()
                total += y.numel()

    elif victim_model_type == "nft":
        nfnet: AutoEncoder = victim["nfnet"]
        clf: nn.Module = victim["clf"]
        latent_dim = nfnet.latent_shape[-1]
        with torch.no_grad():
            for batch in loader:
                batch_start = time.time()
                batch_count += 1
                x, inr, embedding, y, x_adv, inr_adv = batch
                x = x.to(device, non_blocking=True)
                x_adv = x_adv.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                embedding = embedding.to(device, non_blocking=True)
                # Move nested INR tensors to device
                inr = tuple([[t.to(device) for t in j] for j in inr])
                inr_adv = tuple([[t.to(device) for t in j] for j in inr_adv])

                # Apply diffusion defense if enabled (refit mode)
                if diffusion_refit_mode:
                    denoise_start = time.time()
                    # Denoise clean image
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Build embeddings from denoised images via INR refit
                    refit_start = time.time()
                    clean_embedding = builder.build_embeddings(x_clean_denoised, nfnet=nfnet)
                    adv_embedding = builder.build_embeddings(x_adv_denoised, nfnet=nfnet)
                    refit_time = time.time() - refit_start
                    total_refit_time += refit_time

                    inference_start = time.time()
                    clean_logits = clf(clean_embedding.view(x.shape[0], 16, latent_dim))
                    adv_logits = clf(adv_embedding.view(x.shape[0], 16, latent_dim))
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, refit={refit_time:.2f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")
                elif diffusion_surrogate_mode:
                    from src.attacks.surrogate.train_surrogate import unflatten_to_wsfeat

                    denoise_start = time.time()
                    # Denoise clean image
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Use surrogate to predict INR weights from denoised images
                    surrogate_start = time.time()
                    batch_size = x.shape[0]
                    # Clean: Image -> surrogate -> INR weights -> embedding
                    flat_clean = x_clean_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_clean_flat = surrogate(flat_clean)
                    clean_wsfeat = unflatten_to_wsfeat(pred_clean_flat, weight_shapes, bias_shapes, device=device)
                    clean_embedding = builder.build_embeddings_from_wts_and_bs(wts_and_bs=(clean_wsfeat.weights, clean_wsfeat.biases), nfnet=nfnet)

                    # Adversarial: Image -> surrogate -> INR weights -> embedding
                    flat_adv = x_adv_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_adv_flat = surrogate(flat_adv)
                    adv_wsfeat = unflatten_to_wsfeat(pred_adv_flat, weight_shapes, bias_shapes, device=device)
                    adv_embedding = builder.build_embeddings_from_wts_and_bs(wts_and_bs=(adv_wsfeat.weights, adv_wsfeat.biases), nfnet=nfnet)
                    surrogate_time = time.time() - surrogate_start
                    total_refit_time += surrogate_time

                    inference_start = time.time()
                    clean_logits = clf(clean_embedding.view(batch_size, 16, latent_dim))
                    adv_logits = clf(adv_embedding.view(batch_size, 16, latent_dim))
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, surrogate={surrogate_time:.3f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")
                else:
                    # Use pre-computed embeddings/INRs
                    clean_logits = clf(embedding.view(x.shape[0], 16, latent_dim))
                    adv_embedding = builder.build_embeddings_from_wts_and_bs(wts_and_bs=inr_adv, nfnet=nfnet)
                    adv_logits = clf(adv_embedding.view(x.shape[0], 16, latent_dim))

                clean_pred = clean_logits.argmax(dim=1)
                adv_pred = adv_logits.argmax(dim=1)

                clean_correct += (clean_pred == y).sum().item()
                adv_correct += (adv_pred == y).sum().item()
                total += y.numel()

    elif victim_model_type == "scalegmn":
        # ScaleGMN uses torch_geometric batch format
        model = victim["scalegmn"]
        conf = victim["config"]

        # Get batch params from config
        batch_params = get_scalegmn_batch_params(conf)

        with torch.no_grad():
            for batch in loader:
                batch_start = time.time()
                batch_count += 1
                # (x, inr, embedding, y, x_adv, inr_adv)
                x, inr, _, y, x_adv, inr_adv = batch
                x = x.to(device, non_blocking=True)
                x_adv = x_adv.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Move nested INR tensors to device
                inr = tuple([[t.to(device) for t in j] for j in inr])
                inr_adv = tuple([[t.to(device) for t in j] for j in inr_adv])

                # Apply diffusion defense if enabled (refit mode)
                if diffusion_refit_mode:
                    denoise_start = time.time()
                    # Denoise clean image and refit INR
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Refit INRs from denoised images
                    refit_start = time.time()
                    clean_wts_and_bs = builder.build_wts_and_bs_from_img(x_clean_denoised)
                    clean_wsfeat = WeightSpaceFeatures(*clean_wts_and_bs)

                    adv_wts_and_bs = builder.build_wts_and_bs_from_img(x_adv_denoised)
                    adv_wsfeat = WeightSpaceFeatures(*adv_wts_and_bs)
                    refit_time = time.time() - refit_start
                    total_refit_time += refit_time
                elif diffusion_surrogate_mode:
                    from src.attacks.surrogate.train_surrogate import unflatten_to_wsfeat

                    denoise_start = time.time()
                    # Denoise clean image
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Use surrogate to predict INR weights from denoised images
                    surrogate_start = time.time()
                    batch_size = x.shape[0]
                    # Clean: Image -> surrogate -> INR weights
                    flat_clean = x_clean_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_clean_flat = surrogate(flat_clean)
                    clean_wsfeat = unflatten_to_wsfeat(pred_clean_flat, weight_shapes, bias_shapes, device=device)

                    # Adversarial: Image -> surrogate -> INR weights
                    flat_adv = x_adv_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_adv_flat = surrogate(flat_adv)
                    adv_wsfeat = unflatten_to_wsfeat(pred_adv_flat, weight_shapes, bias_shapes, device=device)
                    surrogate_time = time.time() - surrogate_start
                    total_refit_time += surrogate_time
                else:
                    # Use pre-computed INR weights
                    clean_wsfeat = WeightSpaceFeatures(*inr)
                    adv_wsfeat = WeightSpaceFeatures(*inr_adv)

                # Convert to PyG batch format
                inference_start = time.time()
                clean_batch = wsfeat_to_pyg_batch(clean_wsfeat, **batch_params)
                adv_batch = wsfeat_to_pyg_batch(adv_wsfeat, **batch_params)

                clean_batch = clean_batch.to(device)
                adv_batch = adv_batch.to(device)

                # Forward through ScaleGMN
                clean_logits = model(clean_batch)
                adv_logits = model(adv_batch)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                if diffusion_refit_mode:
                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, refit={refit_time:.2f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")
                elif diffusion_surrogate_mode:
                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, surrogate={surrogate_time:.3f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")

                clean_pred = clean_logits.argmax(dim=1)
                adv_pred = adv_logits.argmax(dim=1)
                clean_correct += (clean_pred == y).sum().item()
                adv_correct += (adv_pred == y).sum().item()
                total += y.numel()

    elif victim_model_type == "mwt":
        # MWT processes images end-to-end with inner loop SIREN fitting
        from basemodels.utils.mwt.train import forward as mwt_forward
        from basemodels.utils.mwt.validation import validation
        from basemodels.utils.mwt.util import get_mgrid
        from einops import rearrange

        siren_rgb = victim["siren_rgb"]
        classifier = victim["classifier"]
        meta_sgd = victim["meta_sgd"]
        PI = victim["PI"]
        conf = victim["config"]

        # MWT config for forward pass
        mwt_conf = {
            'inner_steps': 6,
            'is_3d': False,
            'color_channels': conf['color_channels'],
            'sample_size': side * side,
            'cls_loss_weight': 0.0,  # Inference mode
        }

        # with torch.no_grad():
        for batch in tqdm(loader, desc="MWT Evaluation"):
            x, _, _, y, x_adv, _ = batch
            x = x.to(device, non_blocking=True)
            x_adv = x_adv.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Clean forward pass
            clean_logits, _, _ = mwt_forward(
                conf=mwt_conf,
                image=x,
                preview_dict=None,
                build_graph=False,
                device=device,
                PI=PI,
                siren_rgb=siren_rgb,
                classifier=classifier,
                meta_sgd=meta_sgd,
            )

            # Apply diffusion defense if enabled (for MWT, just denoise image)
            if diffusion_refit_mode:
                x_adv_normalized = x_adv * 2 - 1  # Convert to [-1, 1] for denoiser
                x_adv = denoiser(x_adv_normalized)
                x_adv = (x_adv + 1) / 2  # Convert back to [0, 1]

            # Adversarial forward pass
            adv_logits, _, _ = mwt_forward(
                conf=mwt_conf,
                image=x_adv,
                preview_dict=None,
                build_graph=False,
                device=device,
                PI=PI,
                siren_rgb=siren_rgb,
                classifier=classifier,
                meta_sgd=meta_sgd,
                eval_mode=True
            )

            clean_pred = clean_logits.argmax(dim=1)
            adv_pred = adv_logits.argmax(dim=1)
            clean_correct += (clean_pred == y).sum().item()
            adv_correct += (adv_pred == y).sum().item()
            total += y.numel()

    elif victim_model_type == "dws":
        # DWS processes INR weights directly
        model = victim["dws"]
        with torch.no_grad():
            for batch in tqdm(loader, desc="DWS Evaluation"):
                batch_start = time.time()
                batch_count += 1
                x, inr, _, y, x_adv, inr_adv = batch
                x = x.to(device, non_blocking=True)
                x_adv = x_adv.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Move INR tensors to device and convert to WeightSpaceFeatures
                inr = tuple([[t.to(device) for t in j] for j in inr])
                inr_adv = tuple([[t.to(device) for t in j] for j in inr_adv])

                # Apply diffusion defense if enabled (refit mode)
                if diffusion_refit_mode:
                    denoise_start = time.time()
                    # Denoise clean image and refit INR
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Refit INRs from denoised images
                    refit_start = time.time()
                    clean_wts_and_bs = builder.build_wts_and_bs_from_img(x_clean_denoised)
                    clean_wsfeat = WeightSpaceFeatures(*clean_wts_and_bs)

                    adv_wts_and_bs = builder.build_wts_and_bs_from_img(x_adv_denoised)
                    adv_wsfeat = WeightSpaceFeatures(*adv_wts_and_bs)
                    refit_time = time.time() - refit_start
                    total_refit_time += refit_time
                elif diffusion_surrogate_mode:
                    from src.attacks.surrogate.train_surrogate import unflatten_to_wsfeat

                    denoise_start = time.time()
                    # Denoise clean image
                    x_clean_normalized = x * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_clean_denoised = denoiser(x_clean_normalized)
                    x_clean_denoised = (x_clean_denoised + 1) / 2  # Convert back to [0, 1]

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2
                    denoise_time = time.time() - denoise_start
                    total_denoise_time += denoise_time

                    # Use surrogate to predict INR weights from denoised images
                    surrogate_start = time.time()
                    batch_size = x.shape[0]
                    # Clean: Image -> surrogate -> INR weights
                    flat_clean = x_clean_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_clean_flat = surrogate(flat_clean)
                    clean_wsfeat = unflatten_to_wsfeat(pred_clean_flat, weight_shapes, bias_shapes, device=device)

                    # Adversarial: Image -> surrogate -> INR weights
                    flat_adv = x_adv_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_adv_flat = surrogate(flat_adv)
                    adv_wsfeat = unflatten_to_wsfeat(pred_adv_flat, weight_shapes, bias_shapes, device=device)
                    surrogate_time = time.time() - surrogate_start
                    total_refit_time += surrogate_time
                else:
                    # Use pre-computed INR weights
                    clean_wsfeat = WeightSpaceFeatures(*inr)
                    adv_wsfeat = WeightSpaceFeatures(*inr_adv)

                inference_start = time.time()
                clean_weights_dws, clean_biases_dws = wsfeat_to_dws(clean_wsfeat)
                adv_weights_dws, adv_biases_dws = wsfeat_to_dws(adv_wsfeat)

                clean_logits = model((clean_weights_dws, clean_biases_dws))
                adv_logits = model((adv_weights_dws, adv_biases_dws))
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                if diffusion_refit_mode:
                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, refit={refit_time:.2f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")
                elif diffusion_surrogate_mode:
                    batch_time = time.time() - batch_start
                    print(f"  [Eval] Batch {batch_count}: denoise={denoise_time:.2f}s, surrogate={surrogate_time:.3f}s, inference={inference_time:.3f}s, total={batch_time:.2f}s")

                clean_pred = clean_logits.argmax(dim=1)
                adv_pred = adv_logits.argmax(dim=1)
                clean_correct += (clean_pred == y).sum().item()
                adv_correct += (adv_pred == y).sum().item()
                total += y.numel()

    elif victim_model_type == "inr2vec":
        # INR2Vec: INR weights -> matrix -> encoder -> embedding -> classifier -> logits
        encoder = victim["encoder"]
        clf = victim["clf"]
        hidden_dim = victim.get("hidden_dim", 32)

        with torch.no_grad():
            for batch in tqdm(loader, desc="INR2Vec Evaluation"):
                x, inr, _, y, x_adv, inr_adv = batch
                x = x.to(device, non_blocking=True)
                x_adv = x_adv.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Move INR tensors to device
                inr = tuple([[t.to(device) for t in j] for j in inr])
                inr_adv = tuple([[t.to(device) for t in j] for j in inr_adv])

                # Convert INR weights to matrix format (batched, no loop)
                clean_wsfeat = WeightSpaceFeatures(*inr)
                clean_matrices = _wsfeat_to_matrix_batched(clean_wsfeat, hidden_dim).float()

                # Apply diffusion defense if enabled (refit mode)
                if diffusion_refit_mode:
                    x_denoised = denoiser(x_adv)
                    adv_wts_and_bs = builder.build_wts_and_bs_from_img(x_denoised)
                    adv_wsfeat = WeightSpaceFeatures(*adv_wts_and_bs)
                elif diffusion_surrogate_mode:
                    from src.attacks.surrogate.train_surrogate import unflatten_to_wsfeat

                    # Denoise adversarial image
                    x_adv_normalized = x_adv * 2 - 1  # Convert to [-1, 1] for denoiser
                    x_adv_denoised = denoiser(x_adv_normalized)
                    x_adv_denoised = (x_adv_denoised + 1) / 2  # Convert back to [0, 1]

                    # Use surrogate to predict INR weights from denoised image
                    batch_size = x.shape[0]
                    flat_adv = x_adv_denoised.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
                    pred_adv_flat = surrogate(flat_adv)
                    adv_wsfeat = unflatten_to_wsfeat(pred_adv_flat, weight_shapes, bias_shapes, device=device)
                else:
                    adv_wsfeat = WeightSpaceFeatures(*inr_adv)

                adv_matrices = _wsfeat_to_matrix_batched(adv_wsfeat, hidden_dim).float()

                # Forward through encoder and classifier
                clean_embedding = encoder(clean_matrices)
                adv_embedding = encoder(adv_matrices)

                clean_logits = clf(clean_embedding)
                adv_logits = clf(adv_embedding)

                clean_pred = clean_logits.argmax(dim=1)
                adv_pred = adv_logits.argmax(dim=1)
                clean_correct += (clean_pred == y).sum().item()
                adv_correct += (adv_pred == y).sum().item()
                total += y.numel()

    else:
        raise ValueError(f"Evaluation for victim_model={victim_model_type} is not implemented in this pathway.")

    return {
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
        "total": total,
    }


def _wsfeat_to_matrix_batched(wsfeat: WeightSpaceFeatures, hdim: int) -> torch.Tensor:
    """Convert WeightSpaceFeatures to matrix format for INR2Vec encoder (batched).

    Args:
        wsfeat: WeightSpaceFeatures with weights of shape (batch, 1, out, in)
        hdim: Hidden dimension of the INR

    Returns:
        Tensor of shape (batch, num_neurons, hdim+1)
    """
    weights = wsfeat.weights  # tuple of (batch, 1, out, in)
    biases = wsfeat.biases    # tuple of (batch, 1, out)
    num_layers = len(weights)

    if num_layers <= 2:
        # Only input and output layers, flatten all
        params = []
        for w, b in zip(weights, biases):
            # w: (batch, 1, out, in) -> (batch, out*in)
            # b: (batch, 1, out) -> (batch, out)
            params.append(w.squeeze(1).reshape(w.shape[0], -1))
            params.append(b.squeeze(1).reshape(b.shape[0], -1))
        params = torch.cat(params, dim=-1)  # (batch, total_params)
        return params.reshape(params.shape[0], -1, hdim)

    # Skip first and last layer, use middle layers
    layer_params = []
    for i in range(1, num_layers - 1):
        w = weights[i].squeeze(1)  # (batch, out, in)
        b = biases[i].squeeze(1)   # (batch, out)
        # Concatenate weight rows with corresponding bias: (batch, out, in+1)
        lp = torch.cat([w, b.unsqueeze(-1)], dim=-1)
        layer_params.append(lp)

    if len(layer_params) == 0:
        # Fallback: flatten all
        params = []
        for w, b in zip(weights, biases):
            params.append(w.squeeze(1).reshape(w.shape[0], -1))
            params.append(b.squeeze(1).reshape(b.shape[0], -1))
        params = torch.cat(params, dim=-1)
        return params.reshape(params.shape[0], -1, hdim)

    # Concatenate all middle layers: (batch, total_neurons, in+1)
    return torch.cat(layer_params, dim=1)
