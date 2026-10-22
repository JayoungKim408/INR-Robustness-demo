import os
import sys
import torch
from typing import Dict, Any
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader

from attacks.utils import DEFAULT_EPS, DEFAULT_EPS_STD
from attacks.utils import SPLIT_POINTS
from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset
from basemodels.utils.nfn.siren_utils import get_batch_inr
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from basemodels.model_loading import load_victim_model
from defences.AT_framework import ATConfig, AdversarialTraining

def adversarial_training(
    dataset_name: str,
    device: torch.device,
    victim_model: str = "nft",
    at_type: str = "inr_at",
    params: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, str]:
    
    
    # Extract config
    inr_config = kwargs['inr_config']
    path_config = kwargs['path_config']
    wandb_manager = kwargs['wandb_manager']
    siren_path = kwargs.get('siren_path', inr_config.get_inr_path(base_path=path_config.base_path, dataset=dataset_name))

    # Setup directories
    defense_dir = path_config.get_basemodel_dir(model=victim_model, dataset=dataset_name, inr_config=inr_config, at_type=at_type)
    if victim_model in ["nft", "inr2vec"]:
        defense_dir = os.path.join(defense_dir, "classifier")  # Save classifier checkpoint for encoder-based models
    os.makedirs(defense_dir, exist_ok=True)

    train_dataset = get_SirenAdnOriginalDataset(
        siren_path=siren_path,
        siren_prefix=inr_config.prefix,
        data_path=path_config.data_path,
        split="train",
        split_points=SPLIT_POINTS[dataset_name],
    )

    val_dataset = get_SirenAdnOriginalDataset(
        siren_path=siren_path,
        siren_prefix=inr_config.prefix,
        data_path=path_config.data_path,
        split="val",
        split_points=SPLIT_POINTS[dataset_name],
    )
    batch_size = params.get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # AT Config
    at_config = ATConfig(
        eps=params.get('eps', DEFAULT_EPS[dataset_name] if at_type == "inr_at" else DEFAULT_EPS_STD[dataset_name]),
        eps_img=params.get('eps_img', DEFAULT_EPS[dataset_name]),
        steps=params.get('steps', 20),
        alpha=params.get('alpha', None),
        lambda_adv=params.get('lambda_adv', 0.5),
        random_start=params.get('random_start', True),
    )

    if at_config.alpha is None:
        at_config.alpha = 2 * at_config.eps / at_config.steps

    encoder_based = victim_model.split("_")[0] in ["nft", "inr2vec"]
    print(f"Adversarial Training Type: {at_type}, Encoder-based: {encoder_based}")
    victim_wrapped = load_victim_model(
                                    dataset_name=dataset_name,
                                    device=device, 
                                    victim_model=victim_model,
                                    load_state=encoder_based, 
                                    load_state_only_encoder=encoder_based,
                                    return_wrapped=True,
                                    **kwargs)
    
    batched_renderer = get_batch_inr(dataset_name)[0] if at_type == "inr_at" else None
    at_trainer = AdversarialTraining(victim_wrapped, at_config, mode=at_type, renderer=batched_renderer)
    

    OPTIM_MAP = {
        "adam": Adam,
        "adamw": AdamW
    }
    if encoder_based:
        # encoder freeze
        for p in victim_wrapped.model.parameters():
            p.requires_grad_(False)

        # classifier만 optimize
        optimizer = OPTIM_MAP[params['optim']](
            victim_wrapped.clf.parameters(),
            lr=params.get('lr', 1e-4),
            weight_decay=params.get('wd', 0.0),
        )
    else:
        optimizer = OPTIM_MAP[params['optim']](victim_wrapped.parameters(), lr=params.get('lr', 1e-4), weight_decay=params.get('wd', 0.0))
    model_to_save = {'victim_wrapped': victim_wrapped}

    # Training loop
    patience = 10 
    num_epochs = params.get('epochs', 100)
    best_val_adv_acc = 0.0
    total_batches = len(train_loader)
    epochs_without_improvement = 0
    print(f"Starting adversarial training: {at_type} on {victim_model}")
    print(f"AT Config: eps={at_config.eps}, steps={at_config.steps}, lambda_adv={at_config.lambda_adv}")
    print(f"Training for {num_epochs} epochs, {total_batches} batches per epoch")
    print(f"Defense checkpoint will be saved to: {defense_dir}")
    sys.stdout.flush()

    for epoch in range(num_epochs):
        at_trainer.train()
        if encoder_based:
            victim_wrapped.model.eval()

        epoch_metrics = {'total_loss': 0, 'clean_loss': 0, 'adv_loss': 0, 'clean_acc': 0, 'adv_acc': 0}
        num_batches = 0

        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        sys.stdout.flush()

        for batch_idx, batch in enumerate(train_loader):
            # params, img, encoder_B, label = batch
            params, init_params, img, label = batch
            wsfeat = WeightSpaceFeatures(*params).to(device)
            img = img.to(device)
            label = label.to(device)

            # Both EmbeddingSpaceAT and INRSpaceAT use wsfeat-only train_step
            metrics = at_trainer.train_step(wsfeat, label, optimizer)

            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_loss = epoch_metrics['total_loss'] / num_batches
                avg_clean_acc = epoch_metrics['clean_acc'] / num_batches
                avg_adv_acc = epoch_metrics['adv_acc'] / num_batches
                print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {avg_loss:.4f}, Clean: {avg_clean_acc:.4f}, Adv: {avg_adv_acc:.4f}")
                sys.stdout.flush()

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        val_metrics = at_trainer.validation(val_loader)
        sys.stdout.flush()
        epoch_metrics = epoch_metrics | val_metrics
        # Average metrics

        if wandb_manager is not None:
            wandb_manager.log({f"at/{k}": v for k, v in epoch_metrics.items()})
        print(f"Epoch {epoch+1}/{num_epochs} Summary - Loss: {epoch_metrics['total_loss']:.4f}, "
            f"Clean Acc: {epoch_metrics['clean_acc']:.4f}, Adv Acc: {epoch_metrics['adv_acc']:.4f}"
            f"Val Loss: {val_metrics['val_loss']:.4f}, Val Clean: {val_metrics['val_clean_acc']:.4f}, Val Adv: {val_metrics['val_adv_acc']:.4f}")
        sys.stdout.flush()

        # Save best model
        if epoch_metrics['val_adv_acc'] > best_val_adv_acc:
            epochs_without_improvement = 0
            best_val_adv_acc = epoch_metrics['val_adv_acc']
            save_dict = {
                'at_config': at_config,
                'epoch': epoch,
                'adv_acc': best_val_adv_acc,
                'at_type': at_type,
                'victim_model': victim_model,
            }
            for name, model in model_to_save.items():
                save_dict['basemodel'] = model.model.state_dict()
                if encoder_based:
                    save_dict['classifier'] = model.clf.state_dict()
            torch.save(save_dict, os.path.join(defense_dir, "best.pt"))
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    save_dict = {
        'at_config': at_config,
        'epoch': num_epochs - 1,
        'at_type': at_type,
        'victim_model': victim_model,
    }
    for name, model in model_to_save.items():
        save_dict['basemodel'] = model.model.state_dict()
        if encoder_based:
            save_dict['classifier'] = model.clf.state_dict()
    torch.save(save_dict, os.path.join(defense_dir, "last.pt"))

    print(f"Adversarial training completed. Best val adv acc: {best_val_adv_acc:.4f}")
    return os.path.join(defense_dir, "best.pt")
