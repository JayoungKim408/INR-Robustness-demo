"""Training functions for base model classifiers (NFN, NFT, ScaleGMN, MWT, DWS)."""

import os
import sys
import glob
import time
from typing import List, Dict

import torch

from utils.path_config import PathConfig
from utils.config_utils import load_config
from utils.wandb_manager import WandbManager
from attacks.utils import SPLIT_POINTS
# ============================================================
# Source directory path (for accessing config files that are part of source code)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# GPU Management
# ============================================================
def set_visible_gpus(gpu_ids: List[int]):
    """Set visible GPUs for the current process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"Using GPUs: {gpu_ids}")


# ============================================================
# Training Functions for Base Models
# ============================================================
def train_nfn(dataset_name: str, base_path: str, **kwargs) -> str:
    """
    Train NFN (Neural Functional Networks) classifier.

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        base_path: Base path for data and checkpoints
        inr_config: Optional INRConfig for ablation studies
        siren_path: Optional custom siren path
        use_wandb: Whether to use wandb logging

    Returns:
        Path to best checkpoint
    """
    from basemodels.nfn import main as train_nfn_main

    # Extract INR config from kwargs
    inr_config = kwargs.get('inr_config', None)
    path_config = kwargs.get('path_config', None)
    wandb_manager = kwargs.get('wandb_manager', None)

    # Determine save directory (INR-config-aware)
    save_dir = path_config.get_basemodel_dir(model=kwargs.get("victim_model", "nfn"), dataset=dataset_name, inr_config=inr_config)
    os.makedirs(save_dir, exist_ok=True)

    # Load config from basemodels/configs/nfn.yaml
    config_path = os.path.join(SRC_DIR, "basemodels", "configs", "nfn.yaml")
    cfg = load_config(config_path)

    # Override with INR-config-aware paths and dataset
    cfg.data_name = dataset_name
    cfg.data_root_dir = base_path
    cfg.siren_clf_dir = save_dir
    cfg.split_points = SPLIT_POINTS[dataset_name]

    # Override dset paths
    if 'dset' not in cfg:
        cfg['dset'] = {}
    cfg['dset']['siren_path'] = inr_config.get_inr_path(base_path, dataset_name)
    cfg['dset']['siren_prefix'] = inr_config.prefix
    cfg['dset']['split_points'] = SPLIT_POINTS[dataset_name]


    # Override other kwargs if provided
    if 'batch_size' in kwargs:
        cfg.batch_size = kwargs['batch_size']
    if 'max_steps' in kwargs:
        cfg.max_steps = kwargs['max_steps']
    if 'lr' in kwargs:
        cfg.opt.lr = kwargs['lr']
    if 'seed' in kwargs:
        cfg.seed = kwargs['seed']

    cfg.extra_aug = inr_config.extra_aug
    if inr_config.extra_aug:
        cfg.n_extra_aug = 20 if dataset_name == 'cifar' else 10
    else:
        cfg.n_extra_aug = 0

    cfg.hidden_features = inr_config.width
    cfg.hidden_layers = inr_config.depth
    cfg.first_omega_0 = inr_config.scale
    cfg.hidden_omega_0 = inr_config.scale

    # Create wandb manager
    # wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False))

    # Run training
    train_nfn_main(cfg, wandb_manager)

    return os.path.join(save_dir, "best.pt")


def train_nft(dataset_name: str, base_path: str,
              train_encoder: bool = True, train_classifier: bool = True, **kwargs) -> Dict[str, str]:
    """
    Train NFT (Neural Functional Transformer) - 2-stage pipeline.
    Stage 1: Train AutoEncoder (inr2array)
    Stage 2: Train TransformerClassifier

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        base_path: Base path for data and checkpoints
        train_encoder: Whether to train encoder
        train_classifier: Whether to train classifier
        inr_config: Optional INRConfig for ablation studies
        siren_path: Optional custom siren path
        use_wandb: Whether to use wandb logging

    Returns:
        Dict with paths to best checkpoints
    """
    from basemodels.inr2array import train_and_eval as train_inr2array
    from basemodels.nft import main as train_nft_classifier

    # Extract INR config from kwargs
    inr_config = kwargs.get('inr_config', None)
    path_config = kwargs.get('path_config', None)
    siren_path = inr_config.get_inr_path(base_path, dataset_name)

    latent_dim = kwargs.get('latent_dim', 256)

    base_model_dir = path_config.get_basemodel_dir(model=kwargs.get("victim_model", "nft"), 
                                                   dataset=dataset_name, 
                                                   inr_config=inr_config, 
                                                   latent_dim=latent_dim
                                                   )
    encoder_dir = os.path.join(base_model_dir, "encoder")
    classifier_dir = os.path.join(base_model_dir, "classifier")
    embedding_dir = os.path.join(base_model_dir, "embeddings")
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(classifier_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    results = {}

    # Stage 1: Train AutoEncoder (inr2array)
    # Load dataset-specific config
    config_path = os.path.join(SRC_DIR, "basemodels", "configs", f"inr2array_{dataset_name}.yaml")
    cfg = load_config(config_path)

    # Override with INR-config-aware paths
    cfg.dataset_name = dataset_name
    # cfg.ddp_info = ddp_info
    cfg.data_root_dir = base_path
    cfg.inr2array_model_dir = encoder_dir
    cfg.embedding_dir = embedding_dir
    cfg.inr_config = inr_config

    # Override dset paths
    if 'dset' not in cfg:
        cfg['dset'] = {}
    cfg['dset']['siren_path'] = siren_path
    cfg['dset']['siren_prefix'] = inr_config.prefix
    cfg['dset']['data_path'] = path_config.data_path
    cfg['dset']['split_points'] = SPLIT_POINTS[dataset_name]
    
    cfg['model']['inr_config'] = inr_config
    # Override other kwargs
    if 'batch_size' in kwargs:
        cfg.bs = kwargs['batch_size']
    if 'total_steps' in kwargs:
        cfg.total_steps = kwargs['total_steps']
    if 'lr' in kwargs:
        cfg.lr = kwargs['lr']
    if 'latent_dim' in kwargs:
        cfg['pool_cls']['latent_dim'] = latent_dim
    if 'gradient_accumulation_steps' in kwargs:
        cfg.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']

    cfg.extra_aug = inr_config.extra_aug
    if inr_config.extra_aug:
        cfg.n_extra_aug = 20 if dataset_name == 'cifar' else 10
    else: 
        cfg.n_extra_aug = 0 
        
    if train_encoder:
        # Create wandb manager
        # wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False))

        # Run training
        train_inr2array(cfg, wandb_manager=kwargs.get('wandb_manager'), **kwargs.get('ddp_info', {}))
        results["encoder"] = os.path.join(encoder_dir, "best.pt")
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            if kwargs['ddp_info']["rank"] != 0:
                return None
        # import torch.distributed as dist
        # dist.barrier()  # 모든 rank 동기화
        # if kwargs['ddp_info']["rank"] != 0:
        #     return None  # rank 0만 계속 진행
    # Stage 2: Train Classifier
    if train_classifier:
        # Create wandb manager
        # wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False))

        # Load classifier config
        config_path = os.path.join(SRC_DIR, "basemodels", "configs", "nft.yaml")
        clf_cfg = load_config(config_path)
        clf_cfg['inr2array'] = cfg
        clf_cfg['latent_clf_dir'] = classifier_dir
        clf_cfg['inr2array_model_dir'] = encoder_dir

        # Override with paths
        clf_cfg.dataset_name = dataset_name

        # Override other kwargs
        if 'n_epochs' in kwargs:
            clf_cfg.n_epochs = kwargs['n_epochs']
        if 'lr' in kwargs:
            clf_cfg.lr = kwargs['lr']

        # Run training
        train_nft_classifier(clf_cfg, wandb_manager=kwargs.get('wandb_manager'))
        results["classifier"] = os.path.join(classifier_dir, "best.pt")

    return results


def train_scalegmn(dataset_name: str, base_path: str, **kwargs) -> str:
    """
    Train ScaleGMN (Scale-Equivariant Graph MetaNetworks).

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        base_path: Base path for data and checkpoints
        inr_config: Optional INRConfig for ablation studies
        siren_path: Optional custom siren path
        use_wandb: Whether to use wandb logging

    Returns:
        Path to best checkpoint
    """
    from basemodels.scalegmn import main as train_scalegmn_main
    from basemodels.utils.scalegmn.utils.helpers import update_common_args
    # Extract INR config from kwargs
    inr_config = kwargs.get('inr_config', None)
    path_config = kwargs.get('path_config', None)
    siren_path = inr_config.get_inr_path(base_path, dataset_name)
    prefix = inr_config.prefix

    # Determine save directory (INR-config-aware)
    save_dir = path_config.get_basemodel_dir(model=kwargs.get("victim_model", "scalegmn"), dataset=dataset_name, inr_config=inr_config)
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset-specific config from basemodels/configs/
    config_path = os.path.join(SRC_DIR, "basemodels", "configs", f"scalegmn_{dataset_name}.yaml")
    conf = load_config(config_path)

    conf['prefix'] = prefix
    # Override with INR-config-aware paths
    conf['save_dir'] = save_dir
    conf['data']['dataset_path'] = siren_path
    conf['batch_size'] = kwargs.get('batch_size', conf['batch_size'])

    # Update layer_layout if inr_config is provided
    if inr_config is not None:
        # layer_layout is [input_dim, hidden_widths..., output_dim]
        # For SIREN: [2, width, width, ..., 1] with depth hidden layers
        if dataset_name == "shapenet":
            layer_layout = [3] + [inr_config.width] * (inr_config.depth + 1) + [1]
        else:
            layer_layout = [2] + [inr_config.width] * (inr_config.depth + 1) + [1 if dataset_name != 'cifar' else 3]

        conf['data']['layer_layout'] = layer_layout

    conf['extra_aug'] = inr_config.extra_aug
    if inr_config.extra_aug:
        conf['n_extra_aug'] = 20 if dataset_name == 'cifar' else 10
    else: 
        conf['n_extra_aug'] = 0 

    # Override other kwargs
    if 'num_epochs' in kwargs:
        conf['train_args']['num_epochs'] = kwargs['num_epochs']

    # Set wandb
    conf['wandb'] = kwargs.get('use_wandb', False)
    conf = update_common_args(conf)

    # Create wandb manager
    wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False),)

    # Run training
    train_scalegmn_main(conf=conf, wandb_manager=wandb_manager, **kwargs.get('ddp_info', {}))

    return os.path.join(save_dir, "best.pt")


def train_mwt(dataset_name: str, base_path: str,
              epochs: int = 10, batch_size: int = 16, **kwargs) -> str:
    """
    Train MWT (Meta Weight Transformer).

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar10'
        base_path: Base path for data and checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        inr_config: INRConfig object for custom INR architecture
        use_wandb: Whether to use wandb logging

    Returns:
        Path to best checkpoint
    """
    from basemodels.mwt import main as train_mwt_main

    # Extract INR config for architecture params
    inr_config = kwargs.get('inr_config', None)

    # Use PathConfig for INR-config-aware checkpoint path
    path_config = PathConfig(base_path=base_path)

    # MWT uses dataset names: 'mnist', 'fashionmnist', 'cifar10'
    mwt_dataset_map = {
        "mnist": "mnist",
        "fashion": "fashionmnist",
        "cifar": "cifar10",
    }
    mwt_dataset = mwt_dataset_map.get(dataset_name, dataset_name)

    # Generate run name
    if inr_config is not None:
        inr_suffix = f"_d{inr_config.depth}_w{inr_config.width}_s{inr_config.scale}"
    else:
        inr_suffix = ""
    run_name = f"mwt_{mwt_dataset}{inr_suffix}_{time.strftime('%m%d_%H%M%S')}"

    # Get INR-config-aware save directory
    save_dir = path_config.get_basemodel_dir(model="mwt", dataset=dataset_name, inr_config=inr_config)
    os.makedirs(save_dir, exist_ok=True)

    # Load config from basemodels/configs/mwt.yaml
    config_path = os.path.join(SRC_DIR, "basemodels", "configs", "mwt.yaml")
    conf = load_config(config_path)

    # Override with provided parameters
    conf['name'] = run_name
    conf['dataset'] = mwt_dataset
    conf['dataset_name'] = dataset_name
    conf['epochs'] = epochs
    conf['batch_size'] = batch_size
    conf['save_dir'] = save_dir
    conf['base_path'] = base_path

    if 'siren_dim' not in conf:
        conf['siren_dim'] = [conf['dim']] * conf['siren_depth']

    # Override other kwargs
    recognized_args = {
        "lr_siren_rgb", "lr_classifier", "lr_sgd_lrs", "weight_decay",
        "inner_steps", "inner_steps_test", "modulation_scale", "cls_loss_weight",
        "shared_lr", "classifier_depth", "mlp_mult", "subsample_points",
        "augmentations", "stages", "auto_cast", "device"
    }
    for key, value in kwargs.items():
        if key in recognized_args:
            conf[key] = value

    # Set wandb
    conf['wandb'] = kwargs.get('use_wandb', False)

    # Create wandb manager
    wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False))

    # Run training
    train_mwt_main(conf, wandb_manager)

    return os.path.join(save_dir, "best.pt")


def _ensure_dws_splits_json(siren_path: str, dataset_name: str, inr_config: any):
    import json
    import re
    name = inr_config.prefix + "_aug" * inr_config.extra_aug
    splits_file = os.path.join(siren_path, f"{name}_splits.json")

    if os.path.exists(splits_file):
        return

    print(f"Generating {name}_splits.json for {siren_path}...")

    if dataset_name == "shapenet":
        # ShapeNet h5 format: h5_root/{split}/*.h5
        h5_root = os.path.join(siren_path, inr_config.prefix)
        data_split = {"train": {"path": [], "label": []}, 
                      "val": {"path": [], "label": []}, 
                      "test": {"path": [], "label": []}}
        
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(h5_root, split)
            h5_files = sorted(glob.glob(os.path.join(split_dir, "*.h5")),
                            key=lambda x: int(os.path.basename(x).split('.')[0]))
            for h5_path in h5_files:
                with h5py.File(h5_path, "r") as f:
                    label = int(f["class_id"][()])
                data_split[split]["path"].append(h5_path)
                data_split[split]["label"].append(label)
        
        # No extra_aug for shapenet (already augmented in h5)

    else:
        # Original MNIST/CIFAR/Fashion format: {prefix}/{label}s/net{idx}.pth
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"/(\d)s/"
        idx_to_path, idx_to_label = {}, {}

        for sp in glob.glob(f"{siren_path}/{inr_config.prefix}/[0-9]s/*.pth"):
            idx = int(re.search(idx_pattern, sp).group(1))
            idx_to_path[idx] = sp
            label = int(re.search(label_pattern, sp).group(1))
            idx_to_label[idx] = label

        val_point = 45000 if dataset_name == "cifar" else 55000
        test_point = 50000 if dataset_name == "cifar" else 60000

        idx = {
            "train": list(range(val_point)),
            "val": list(range(val_point, test_point)),
            "test": list(range(test_point, len(idx_to_path))),
        }

        data_split = {"train": {"path": [], "label": []}, 
                      "val": {"path": [], "label": []}, 
                      "test": {"path": [], "label": []}}
        for split in ["train", "val", "test"]:
            data_split[split]['path'] = [idx_to_path[i] for i in idx[split]]
            data_split[split]['label'] = [idx_to_label[i] for i in idx[split]]

        if inr_config.extra_aug:
            n_extra_aug = 20 if dataset_name == "cifar" else 10
            for i in range(1, n_extra_aug):
                extra_idx_to_path, extra_idx_to_label = {}, {}
                for sp in glob.glob(f"{siren_path}/{inr_config.prefix}_{i}/[0-9]s/*.pth"):
                    idx = int(re.search(idx_pattern, sp).group(1))
                    extra_idx_to_path[idx] = sp
                    label = int(re.search(label_pattern, sp).group(1))
                    extra_idx_to_label[idx] = label
                data_split['train']['path'].extend([extra_idx_to_path[i] for i in idx['train']])
                data_split['train']['label'].extend([extra_idx_to_label[i] for i in idx['train']])

    with open(splits_file, "w") as f:
        json.dump(data_split, f, indent=2)

    if dataset_name == "shapenet":
        param_shapes_path = os.path.join(siren_path, "param_shapes.pth")
        if not os.path.exists(param_shapes_path):
            from basemodels.utils.nfn.siren_utils import Siren
            model = Siren(in_features=3, out_features=1, hidden_features=128,
                        hidden_layers=1, outermost_linear=True)
            torch.save([(k, v.shape) for k, v in model.state_dict().items()], param_shapes_path)
            
    print(f"Generated {name}_splits.json with {len(data_split['train']['path'])} train, "
          f"{len(data_split['val']['path'])} val, {len(data_split['test']['path'])} test samples")


def _ensure_dws_statistics(siren_path: str, inr_config: any):
    """
    Ensure statistics.pth exists for DWS training. Generate if not present.

    Args:
        siren_path: Path to SIREN weights directory (must have splits.json)
        prefix: INR prefix ('randinit_smaller' for SIREN, 'ffn' for FFN)
    """
    import json
    name = inr_config.prefix + "_aug" * inr_config.extra_aug
    statistics_file = os.path.join(siren_path, f"{name}_statistics.pth")

    if os.path.exists(statistics_file):
        return  # Already exists

    splits_file = os.path.join(siren_path, f"{name}_splits.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"{name}_splits.json not found at {siren_path}. Run _ensure_dws_splits_json first.")

    print(f"Generating statistics for {siren_path}...")

    # Load splits.json
    with open(splits_file, "r") as f:
        data_split = json.load(f)

    # Get train paths
    train_paths = data_split.get("train", {}).get("path", [])
    if not train_paths:
        raise ValueError(f"No training samples found in {splits_file}")

    # Limit number of samples for efficiency
    sample_paths = train_paths

    # Collect weights and biases from all samples
    from basemodels.utils.dws.experiments.data import Batch, INRDataset
    train_set = INRDataset(path=splits_file, split="train", normalize=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=10000, shuffle=True, num_workers=4
    )

    batch: Batch = next(iter(train_loader))
    weights_mean = [w.mean(0) for w in batch.weights]
    weights_std = [w.std(0) for w in batch.weights]
    biases_mean = [w.mean(0) for w in batch.biases]
    biases_std = [w.std(0) for w in batch.biases]

    statistics = {
        "weights": {"mean": weights_mean, "std": weights_std},
        "biases": {"mean": biases_mean, "std": biases_std},
    }

    # Save statistics.pth
    torch.save(statistics, statistics_file)
    print(f"Generated {name}_statistics.pth with {len(sample_paths)} samples, {len(weights_mean)} layers")


def train_dws(dataset_name: str, base_path: str,
              epochs: int = 100, batch_size: int = 512, **kwargs) -> str:
    """
    Train DWS (Deep Weight Space) classifier.

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        base_path: Base path for data and checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        inr_config: INRConfig object for custom INR architecture
        siren_path: Path to SIREN weights directory (overrides inr_config path)
        use_wandb: Whether to use wandb logging

    Returns:
        Path to best checkpoint
    """
    from basemodels.dws import main as train_dws_main

    # Extract INR config for INR-config-aware paths
    inr_config = kwargs.get('inr_config', None)
    siren_path = inr_config.get_inr_path(base_path, dataset_name)
    path_config = kwargs.get('path_config', None)

    if siren_path is not None:
        _ensure_dws_splits_json(siren_path, dataset_name, inr_config=inr_config)
        _ensure_dws_statistics(siren_path, inr_config=inr_config)

    # Get INR-config-aware save directory
    save_dir = path_config.get_basemodel_dir(model=kwargs.get("victim_model", "dws"), dataset=dataset_name, inr_config=inr_config)
    os.makedirs(save_dir, exist_ok=True)

    # Load config from basemodels/configs/dws.yaml
    config_path = os.path.join(SRC_DIR, "basemodels", "configs", "dws.yaml")
    conf = load_config(config_path)
    
    conf['extra_aug'] = inr_config.extra_aug
    if inr_config.extra_aug:
        conf['n_extra_aug'] = 20 if dataset_name == 'cifar' else 10
    else: 
        conf['n_extra_aug'] = 0

    # Override with provided parameters
    conf['dataset_name'] = dataset_name
    conf['base_path'] = base_path
    conf['n_epochs'] = epochs
    conf['batch_size'] = batch_size
    conf['save_path'] = save_dir
    conf['siren_path'] = siren_path
    conf['prefix'] = inr_config.prefix

    # Override with kwargs (convert key formats)
    kwarg_mapping = {
        'lr': 'lr',
        'eval_every': 'eval_every',
        'normalize': 'normalize',
        'augmentation': 'augmentation',
        'permutation': 'permutation',
        'n_samples': 'n_samples',
        'num_workers': 'num_workers',
        'model_type': 'model_type',
        'dim_hidden': 'dim_hidden',
        'n_hidden': 'n_hidden',
        'reduction': 'reduction',
        'n_fc_layers': 'n_fc_layers',
        'set_layer': 'set_layer',
        'n_out_fc': 'n_out_fc',
        'do_rate': 'do_rate',
        'add_bn': 'add_bn',
        'diagonal': 'diagonal',
        'optim': 'optim',
        'seed': 'seed',
        'gpu': 'gpu',
    }
    for kwarg_key, conf_key in kwarg_mapping.items():
        if kwarg_key in kwargs:
            conf[conf_key] = kwargs[kwarg_key]

    # Set sensible defaults if not in kwargs (important for DWS)
    if 'num_workers' not in kwargs:
        conf['num_workers'] = 0  # Must be 0 to avoid pickling issues with module imports

    # Set wandb
    conf['wandb'] = kwargs.get('use_wandb', False)

    # Create wandb manager
    wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False))

    # Run training
    train_dws_main(conf, wandb_manager)

    return os.path.join(save_dir, "best.pt")

def unwrap_nested_dictconfig(cfg):
    from utils.config_utils import DictConfig
    if isinstance(cfg, DictConfig):
        # _data 안에 _data가 있으면 unwrap
        if '_data' in cfg._data:
            return unwrap_nested_dictconfig(cfg._data['_data'])
        # 정상적인 DictConfig면 내부 값들도 재귀적으로 처리
        result = {}
        for key, value in cfg._data.items():
            result[key] = unwrap_nested_dictconfig(value)
        return DictConfig(result)
    elif isinstance(cfg, list):
        return [unwrap_nested_dictconfig(item) for item in cfg]
    else:
        return cfg


def train_inr2vec(dataset_name: str, base_path: str,
                  train_encoder: bool = True, train_classifier: bool = True, **kwargs) -> Dict[str, str]:
    """
    Train INR2Vec - 2-stage pipeline for 2D image INRs.
    Stage 1: Train Encoder-Decoder (inr2vec)
    Stage 2: Train Classifier on embeddings

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        base_path: Base path for data and checkpoints
        train_encoder: Whether to train encoder-decoder
        train_classifier: Whether to train classifier
        inr_config: Optional INRConfig for ablation studies
        siren_path: Optional custom siren path
        use_wandb: Whether to use wandb logging
        wandb_manager: Optional WandbManager for logging

    Returns:
        Dict with paths to best checkpoints
    """
    if dataset_name == 'shapenet':
        from basemodels.inr2vec.pcd_retrieval_and_classification.train_inr2vec import Inr2vecTrainer
        from basemodels.inr2vec.pcd_retrieval_and_classification.export_embeddings import export_embeddings
        from basemodels.inr2vec.pcd_retrieval_and_classification.train_classifier import EmbeddingClassifier
        config_path = os.path.join(SRC_DIR, "basemodels", "inr2vec", "pcd_retrieval_and_classification", "cfg")
    else:
        from basemodels.inr2vec.image_classification.train_inr2vec import Inr2vecTrainer
        from basemodels.inr2vec.image_classification.export_embeddings import export_embeddings
        from basemodels.inr2vec.image_classification.train_classifier import EmbeddingClassifier
        config_path = os.path.join(SRC_DIR, "basemodels", "inr2vec", "image_classification", "cfg")

    # Extract INR config from kwargs
    inr_config = kwargs.get('inr_config', None)
    siren_path = kwargs.get('siren_path', inr_config.get_inr_path(base_path, dataset_name))
    wandb_manager = kwargs.get('wandb_manager', None)

    # Use PathConfig for INR-config-aware paths
    path_config = PathConfig(base_path=base_path)

    # Determine save directories (INR-config-aware)
    base_model_dir = path_config.get_basemodel_dir(
        model=kwargs.get("victim_model", "inr2vec"), dataset=dataset_name, inr_config=inr_config
    )
    encoder_dir = os.path.join(base_model_dir, "encoder")
    classifier_dir = os.path.join(base_model_dir, "classifier")
    embeddings_dir = os.path.join(base_model_dir, "embeddings")
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(classifier_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    results = {}

    if train_encoder:

        encoder_config = load_config(os.path.join(config_path, "inr2vec.yaml"))
        if hasattr(encoder_config, '_data'):
            encoder_config = unwrap_nested_dictconfig(encoder_config)

        encoder_config["dataset_name"] = dataset_name
        encoder_config["siren_path"] = siren_path
        encoder_config["siren_prefix"] = inr_config.prefix
        encoder_config["split_points"] = SPLIT_POINTS[dataset_name]
        encoder_config["data_path"] = path_config.data_path
        encoder_config['mlp']['hidden_dim'] = inr_config.width
        encoder_config['mlp']['num_hidden_layers'] = inr_config.depth
        encoder_config['extra_aug'] = inr_config.extra_aug
        encoder_config['encoder']['embedding_dim'] = kwargs.get("latent_dim", 256)
        encoder_config['lr'] = kwargs.get("lr", encoder_config['lr'])
        encoder_config['out_dir'] = encoder_dir

        if inr_config.extra_aug:
            encoder_config['n_extra_aug'] = 20 if dataset_name == 'cifar' else 10
        else: 
            encoder_config['n_extra_aug'] = 0 

        # Create and run trainer with wandb support
        trainer = Inr2vecTrainer(encoder_config, wandb_manager=wandb_manager)
        trainer.train()

    results["encoder"] = os.path.join(encoder_dir, "best.pt")

    # Stage 2: Train Classifier
    if train_classifier:
        make_embedding = False
        if dataset_name == "shapenet":
            embeddings_path = embeddings_dir
            make_embedding = not os.path.join(embeddings_dir, "train")
        else:
            embeddings_path = results.get("embeddings", os.path.join(embeddings_dir, "embeddings.pt"))
            make_embedding = not os.path.exists(embeddings_path)

        if make_embedding:
            # Export embeddings after encoder training
            export_config = load_config(os.path.join(config_path, "export_embeddings.yaml"))
            if hasattr(export_config, '_data'):
                export_config = unwrap_nested_dictconfig(export_config)

            export_config['ckpt_path'] = results["encoder"]
            export_config['out_path'] = embeddings_dir
            export_config['extra_aug'] = inr_config.extra_aug
            export_config['siren_prefix'] = inr_config.prefix

            if inr_config.extra_aug:
                export_config['n_extra_aug'] = 20 if dataset_name == 'cifar' else 10
            else: 
                export_config['n_extra_aug'] = 0 

            export_embeddings(export_config)
        results["embeddings"] = embeddings_dir

        classifier_config = load_config(os.path.join(config_path, "classifier.yaml"))
        if hasattr(classifier_config, '_data'):
            classifier_config = unwrap_nested_dictconfig(classifier_config)

        classifier_config["embeddings_path"] = embeddings_path
        classifier_config["out_dir"] = classifier_dir

        classifier_config['siren_prefix'] = inr_config.prefix
        classifier_config['extra_aug'] = inr_config.extra_aug
        if inr_config.extra_aug:
            classifier_config['n_extra_aug'] = 20 if dataset_name == 'cifar' else 10
        else: 
            classifier_config['n_extra_aug'] = 0 

        # Create and run classifier trainer with wandb support
        clf_trainer = EmbeddingClassifier(classifier_config, wandb_manager=wandb_manager)
        clf_trainer.train()

        results["classifier"] = os.path.join(classifier_dir, "best.pt")

    return results

def train_mlp(dataset_name: str, base_path: str, **kwargs) -> str:
    from basemodels.mlp import main as train_mlp_main
    # Load config from basemodels/configs/mlp.yaml
    config_path = os.path.join(SRC_DIR, "basemodels", "configs", "mlp.yaml")
    cfg = load_config(config_path)

    # Extract INR config from kwargs
    inr_config = kwargs.get('inr_config', None)
    path_config = kwargs.get('path_config', None)
    
    # Determine save directory (INR-config-aware)
    save_dir = path_config.get_basemodel_dir(model=kwargs.get('victim_model'), dataset=dataset_name, inr_config=inr_config)
    os.makedirs(save_dir, exist_ok=True)
    cfg.save_dir = save_dir

    cfg['dset'] = {}
    cfg['dset']['siren_path'] = inr_config.get_inr_path(base_path, dataset_name)
    cfg['dset']['siren_prefix'] = inr_config.prefix
    cfg['dset']['split_points'] = SPLIT_POINTS[dataset_name]

    # Override other kwargs if provided
    if 'batch_size' in kwargs:
        cfg.batch_size = kwargs['batch_size']
    if 'epochs' in kwargs:
        cfg.epochs = kwargs['epochs']
    if 'lr' in kwargs:
        cfg.lr = kwargs['lr']

    if inr_config.extra_aug:
        cfg.extra_aug = inr_config.extra_aug
        cfg.n_extra_aug = 20 if dataset_name == 'cifar' else 10
    else:
        cfg.n_extra_aug = 0 

    # Create wandb manager
    wandb_manager = WandbManager(enabled=kwargs.get('use_wandb', False))

    # Run training
    train_mlp_main(cfg, wandb_manager)

    return os.path.join(save_dir, "best.pt")
