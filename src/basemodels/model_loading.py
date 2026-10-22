"""Victim model loading for different classifier architectures."""

import os
import sys
import glob
import yaml
import torch
import torch.nn as nn
from functools import partial
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

from basemodels.nft import TransformerClassifier
from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from basemodels.inr2array import AutoEncoder, PerceiverPooling
from basemodels.utils.nfn.models import InvariantNFN
from basemodels.utils.dws.nn.models import DWSModelForClassification
from basemodels.victim_wrapper import VictimWrapper
from utils.path_config import PathConfig
from utils.config_utils import load_config
from attacks.utils import wsfeat_to_dws
# from defences.adversarial_training import ATConfig
from defences.AT_framework import ATConfig
# ---------------------------------------------------------------------------
# Wrapper nn.Modules (used by attack methods for INR-space PGD)
# ---------------------------------------------------------------------------

class NFTINRWrapper(nn.Module):
    """Wraps NFT's AutoEncoder + TransformerClassifier for INR-space attacks."""

    def __init__(self, nfnet: AutoEncoder, clf: nn.Module, embedding_wrapper: bool = False):
        super().__init__()
        self.nfnet = nfnet   # inr -> embedding
        self.clf = clf        # embedding -> logit
        self.embedding_wrapper = embedding_wrapper

    def forward(self, input) -> torch.Tensor:
        if self.embedding_wrapper:
            if input.dim() == 2:
                input = input.view(input.shape[0], 16, -1)
            return self.clf(input)
        else:
            emb = self.nfnet.encode(input)
            return self.clf(emb.view(emb.shape[0], 16, 256))


class DWSINRWrapper(nn.Module):
    """Wraps a DWS model to accept WeightSpaceFeatures input."""

    def __init__(self, dws_model):
        super().__init__()
        self.dws_model = dws_model

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        weights, biases = wsfeat_to_dws(wsfeat)
        return self.dws_model((weights, biases))


class INR2VecWrapper(nn.Module):
    """Wrapper for INR2Vec model to work with PGDWSFeat attack."""

    def __init__(self, encoder: nn.Module, clf: nn.Module, hidden_dim: int = 32, embedding_wrapper: bool = False):
        super().__init__()
        self.encoder = encoder
        self.clf = clf
        self.hidden_dim = hidden_dim
        self.embedding_wrapper = embedding_wrapper

    def _wsfeat_to_matrix(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        """Convert WeightSpaceFeatures to matrix format for INR2Vec encoder."""
        weights = wsfeat.weights
        biases = wsfeat.biases
        num_layers = len(weights)
        hdim = self.hidden_dim

        if num_layers <= 2:
            params = []
            for w, b in zip(weights, biases):
                params.append(w.squeeze(1).reshape(w.shape[0], -1))
                params.append(b.squeeze(1).reshape(b.shape[0], -1))
            params = torch.cat(params, dim=-1)
            return params.reshape(params.shape[0], -1, hdim)

        layer_params = []
        for i in range(1, num_layers - 1):
            w = weights[i].squeeze(1)
            b = biases[i].squeeze(1)
            lp = torch.cat([w, b.unsqueeze(-1)], dim=-1)
            layer_params.append(lp)

        if len(layer_params) == 0:
            params = []
            for w, b in zip(weights, biases):
                params.append(w.squeeze(1).reshape(w.shape[0], -1))
                params.append(b.squeeze(1).reshape(b.shape[0], -1))
            params = torch.cat(params, dim=-1)
            return params.reshape(params.shape[0], -1, hdim)

        return torch.cat(layer_params, dim=1)

    def forward(self, input) -> torch.Tensor:
        if self.embedding_wrapper:
            if input.dim() == 3:
                input = input.view(input.shape[0], -1)
            return self.clf(input)
        else:
            matrices = self._wsfeat_to_matrix(input).float()
            embedding = self.encoder(matrices)
            return self.clf(embedding)


# ---------------------------------------------------------------------------
# Helper: infer WeightSpaceFeatures source from a batch
# ---------------------------------------------------------------------------

def _infer_ws_src(first_batch):
    """Find the first tuple/list in the batch that corresponds to siren params."""
    ws_src = None
    if isinstance(first_batch, (list, tuple)):
        for item in first_batch:
            if isinstance(item, (list, tuple)):
                ws_src = item
                break
    if ws_src is None:
        raise ValueError("Could not infer WeightSpaceFeatures source from batch.")
    return ws_src


# ---------------------------------------------------------------------------
# Main loading function (used by Attacks class during attack execution)
# ---------------------------------------------------------------------------
def load_testloader(
    siren_path,
    siren_prefix,
    data_path,
    embedding_path,
    dataset_name
) -> DataLoader:
    
    if embedding_path:
        from basemodels.utils.nfn.data_utils import get_SirenOriginalAndEmbeddingDataset
        dset = get_SirenOriginalAndEmbeddingDataset(siren_path=siren_path,
                                        siren_prefix=siren_prefix,
                                        data_path=data_path,
                                        embedding_path=embedding_path,
                                        split="test",
                                        split_points=[45000, 50000] if dataset_name == "cifar" else [55000, 60000],
                                        )
    else:
        from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset
        dset = get_SirenAdnOriginalDataset(siren_path=siren_path,
                                        siren_prefix=siren_prefix,
                                        data_path=data_path,
                                        split="test",
                                        split_points=[45000, 50000] if dataset_name == "cifar" else [55000, 60000],
                                        )
        
    return DataLoader(dset, batch_size=256, shuffle=False, num_workers=2, drop_last=False)

def load_transfer(dataset_name, device='cuda', **kwargs):
    from attacks.transfer.train_transfer_model.resnet import ResNet18
        
    transfernet = ResNet18(in_c=3 if dataset_name == "cifar" else 1).to(device)
    tn_ckpt = torch.load(os.path.join(kwargs['exp_ckpt_path'], f"{dataset_name}.pth"), map_location="cpu", weights_only=False)["net"]
    tn_ckpt = {k.replace("module.", ""): v for k, v in tn_ckpt.items()}
    transfernet.load_state_dict(tn_ckpt)
    transfernet.eval()
    return transfernet

def load_victim_model(
    victim_model: str,
    dataset_name: str,
    path_config: PathConfig,
    inr_config,
    device: torch.device,
    ckpt_path: Optional[Dict[str, str]] = None,
    testloader: Optional[DataLoader] = None,
    load_state: bool = True,
    load_state_only_encoder: bool = False,
    return_wrapped: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    
    if ckpt_path is None:
        ckpt_path = path_config.get_basemodel_checkpoint(
            model=victim_model, 
            dataset=dataset_name, 
            inr_config=inr_config, 
            latent_dim=kwargs.get("latent_dim", 256), 
        )
    
    if testloader is None and victim_model not in ['scalegmn', 'mwt']:
        testloader = load_testloader(siren_path=inr_config.get_inr_path(path_config.base_path, dataset_name),
                                     siren_prefix=inr_config.prefix, 
                                     data_path=path_config.data_path, embedding_path=None, dataset_name=dataset_name)
        
    if victim_model.startswith("nft"):
        config_path = os.path.join(path_config.base_path, "src", "basemodels", "configs", f"inr2array_{dataset_name}.yaml")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg['inr_config'] = {
            'hidden_layers': inr_config.depth,
            'hidden_features': inr_config.width,
            'first_omega_0': inr_config.scale,
            'hidden_omega_0': inr_config.scale,
        }
        first_batch = next(iter(testloader))
        del testloader
        ws_src = _infer_ws_src(first_batch)
        spec = network_spec_from_wsfeat(WeightSpaceFeatures(*ws_src).to("cpu"), set_all_dims=True)

        model_cfg = dict(cfg.get('model', {})) if cfg.get('model') else {}
        pool_cfg_raw = model_cfg.pop('pool_cls', None) or cfg.get('pool_cls', None)
        if pool_cfg_raw is not None:
            pool_cfg = dict(pool_cfg_raw) if hasattr(pool_cfg_raw, 'items') else {}
            pool_cfg.pop('_target_', None)
            pool_cfg.pop('_partial_', None)
        else:
            pool_cfg = {}
        model_cfg.pop('_target_', None)
        if model_cfg.get('spatial', False) and pool_cfg.get('reduce', False):
            pool_cfg['reduce'] = False

        pool_cfg['latent_dim'] = kwargs.get('latent_dim', 256)
        model_cfg['siren_config'] = cfg['inr_config']
        nfnet = AutoEncoder(model_cfg, spec, dset_data_type=dataset_name,
                            pool_cls=partial(PerceiverPooling, **pool_cfg) if pool_cfg else None
                            ).to(device)
        latent_dim = nfnet.latent_shape[-1]
        clf = TransformerClassifier(latent_dim=latent_dim).to(device)

        nfnet_ckpt, clf_ckpt = ckpt_path['basemodel'], ckpt_path['classifier']

        if load_state and os.path.exists(nfnet_ckpt):
            nfnet.load_state_dict(torch.load(nfnet_ckpt, weights_only=False))
            nfnet.eval()
        else:
            nfnet.train()

        if load_state and os.path.exists(clf_ckpt) and not load_state_only_encoder:
            ckpt = torch.load(clf_ckpt, weights_only=False)
            state = ckpt["clf"] if "clf" in ckpt else ckpt["classifier"]
            clf.load_state_dict(state)
            clf.eval()
        else:
            clf.train()
        model_dict = {"nfnet": nfnet, "clf": clf}
        return VictimWrapper("nft", model_dict, device=device, attack_space=kwargs['attack_space']) if return_wrapped else model_dict


    if victim_model.startswith("nfn"):
        config_path = os.path.join(path_config.base_path, "src", "basemodels", "configs", "nfn.yaml")
        cfg = load_config(config_path)
        first_batch = next(iter(testloader))
        del testloader
        ws_src = _infer_ws_src(first_batch)
        spec = network_spec_from_wsfeat(WeightSpaceFeatures(*ws_src).to("cpu"), set_all_dims=True)

        nfnet_cfg = cfg.get('nfnet', {})
        hchannels = nfnet_cfg.get('hchannels', [512, 512, 512])
        mode = nfnet_cfg.get('mode', 'NP')

        head_cfg = nfnet_cfg.get('head_cls', nfnet_cfg.get('head', {}))
        from basemodels.utils.nfn.models import MlpHead
        head_cls = partial(
            MlpHead,
            num_out=head_cfg.get('num_out', 10),
            dropout=head_cfg.get('dropout', 0.5),
            lnorm=head_cfg.get('lnorm', False),
            sigmoid=head_cfg.get('sigmoid', False),
            pool_mode=mode,
        )

        inp_enc_cfg = nfnet_cfg.get('inp_enc_cls', nfnet_cfg.get('inp_enc', None))
        from basemodels.utils.nfn.nfn.layers import GaussianFourierFeatureTransform, IOSinusoidalEncoding
        inp_enc_cls = partial(
            GaussianFourierFeatureTransform,
            mapping_size=inp_enc_cfg.get('mapping_size', 128) if inp_enc_cfg else 128,
            scale=inp_enc_cfg.get('scale', 10) if inp_enc_cfg else 10,
        ) if inp_enc_cfg else None

        pos_enc_cfg = nfnet_cfg.get('pos_enc_cls', nfnet_cfg.get('pos_enc', None))
        pos_enc_cls = partial(
            IOSinusoidalEncoding,
            max_freq=pos_enc_cfg.get('max_freq', 10) if pos_enc_cfg else 10,
            num_bands=pos_enc_cfg.get('num_bands', 6) if pos_enc_cfg else 6,
        ) if pos_enc_cfg else None

        nfnet = InvariantNFN(
            network_spec=spec,
            hchannels=hchannels,
            head_cls=head_cls,
            mode=mode,
            feature_dropout=nfnet_cfg.get('feature_dropout', 0.0),
            normalize=nfnet_cfg.get('normalize', False),
            lnorm=nfnet_cfg.get('lnorm', None),
            append_stats=nfnet_cfg.get('append_stats', False),
            inp_enc_cls=inp_enc_cls,
            pos_enc_cls=pos_enc_cls,
        ).to(device)

        nfn_ckpt = ckpt_path['basemodel']
        if load_state and os.path.exists(nfn_ckpt):
            state = torch.load(nfn_ckpt, weights_only=False)
            if "at_config" in state: # at checkpoint
                state_dict = state['basemodel']
            else:
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            nfnet.load_state_dict(state_dict)
            nfnet.eval()
        else:
            nfnet.train()

        model_dict = {"nfnet": nfnet}
        return VictimWrapper("nfn", model_dict, device=device, attack_space=kwargs['attack_space']) if return_wrapped else model_dict

    if victim_model.startswith("mlp"):
        from basemodels.mlp import MLPClassifier
        from attacks.surrogate.utils import get_inr_param_count_from_shapes

        config_path = os.path.join(path_config.base_path, "src", "basemodels", "configs", "mlp.yaml")
        cfg = load_config(config_path)
        first_batch = next(iter(testloader))
        sample_wsfeat = WeightSpaceFeatures(*first_batch[0]).to("cpu")
        weight_shapes = [tuple(w.shape[1:]) for w in sample_wsfeat.weights]
        bias_shapes = [tuple(b.shape[1:]) for b in sample_wsfeat.biases]
        total_params = get_inr_param_count_from_shapes(weight_shapes, bias_shapes)

        classifier = MLPClassifier(
            input_dim=total_params,
            num_classes=10,
            hidden_dim=cfg.get('hidden_dim', 1000),
            num_layers=cfg.get('num_layers', 3)
        ).to(device)

        mlp_ckpt = ckpt_path['basemodel']
        if load_state and os.path.exists(mlp_ckpt):
            state = torch.load(mlp_ckpt, weights_only=False)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            classifier.load_state_dict(state_dict)
            classifier.eval()
        else:
            classifier.train()

        model_dict = {"classifier": classifier}
        return VictimWrapper("mlp", model_dict, device=device, attack_space=kwargs['attack_space']) if return_wrapped else model_dict

    if victim_model.startswith("scalegmn"):
        from basemodels.utils.scalegmn.scalegmn.models import ScaleGMN
        from basemodels.utils.scalegmn.utils.helpers import update_common_args

        config_path = os.path.join(path_config.base_path, "src", "basemodels", "configs", f"scalegmn_{dataset_name}.yaml")
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)

        if inr_config is not None:
            layer_layout = [2 if dataset_name != "shapenet" else 3] + [inr_config.width] * (inr_config.depth + 1) + [1 if dataset_name != 'cifar' else 3]
            conf['data']['layer_layout'] = layer_layout

        conf['scalegmn_args']['layer_layout'] = conf['data']['layer_layout']
        conf = update_common_args(conf)

        model = ScaleGMN(conf['scalegmn_args']).to(device)

        basemodel = ckpt_path['basemodel']
        if load_state and os.path.exists(basemodel):
            state_dict = torch.load(basemodel, weights_only=False, map_location=device)
            if state_dict.get("victim_wrapped", None):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict['victim_wrapped'].items()}
            model.load_state_dict(state_dict)
            print(f"Loaded ScaleGMN checkpoint from {basemodel}")
            model.eval()
        else:
            print(f"Warning: ScaleGMN checkpoint not found at {basemodel}")
            model.train()

        model_dict = {"scalegmn": model, "config": conf}
        return VictimWrapper("scalegmn", model_dict, device=device, attack_space=kwargs['attack_space']) if return_wrapped else model_dict


    if victim_model == "mwt":
        from basemodels.utils.mwt.wt import WT
        from basemodels.utils.mwt.siren_model import SirenModel
        from basemodels.utils.mwt.meta_sgd import MetaSGD

        mwt_dataset_map = {"mnist": "mnist", "fashion": "fashionmnist", "cifar": "cifar10"}
        mwt_dataset = mwt_dataset_map.get(dataset_name, dataset_name)

        mwt_config = {
            "mnist": {"siren_dim": [128, 128, 128, 128], "color_channels": 1, "class_count": 10, "dim": 128, "classifier_depth": 10, "mlp_mult": 1, "modulation_scale": 500},
            "fashionmnist": {"siren_dim": [128, 128, 128, 128], "color_channels": 1, "class_count": 10, "dim": 128, "classifier_depth": 10, "mlp_mult": 1, "modulation_scale": 500},
            "cifar10": {"siren_dim": [128, 128, 128, 128], "color_channels": 3, "class_count": 10, "dim": 128, "classifier_depth": 10, "mlp_mult": 1, "modulation_scale": 500},
        }
        conf = mwt_config.get(mwt_dataset, mwt_config["mnist"])
        conf['nfn_channels'] = [conf['dim']] * conf['classifier_depth']
        conf['model'] = 'WT'

        siren_rgb = SirenModel(ch_in=2, ch_hiddens=conf['siren_dim'], ch_out=conf['color_channels'], omega=10.0).to(device)
        PI = siren_rgb.pass_indices.shape[0]

        meta_sgd_conf = {'inner_steps': 6, 'modulation_scale': 500, 'shared_lr': False}
        meta_sgd = MetaSGD(meta_sgd_conf, PI).to(device)

        D = conf['siren_dim'][1]
        classifier = WT(dim=D, conf=conf).to(device)

        if load_state:
            checkpoint_dir = ckpt_path['basemodel']
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
            if checkpoint_files:
                latest_ckpt = max(checkpoint_files, key=os.path.getmtime)
                checkpoint = torch.load(latest_ckpt, weights_only=False, map_location=device)
                siren_rgb.load_state_dict(checkpoint['siren_rgb'])
                classifier.load_state_dict(checkpoint['classifier'])
                meta_sgd.load_state_dict(checkpoint['meta_sgd'])
                print(f"Loaded MWT checkpoint from {latest_ckpt}")
            else:
                print(f"Warning: MWT checkpoint not found in {checkpoint_dir}")
            siren_rgb.eval()
            classifier.eval()
        else:
            siren_rgb.train()
            classifier.train()
            meta_sgd.train()

        model_dict = {
            "siren_rgb": siren_rgb, "classifier": classifier, "meta_sgd": meta_sgd,
            "PI": PI, "config": conf,
        }
        return model_dict

    if victim_model.startswith("dws"):
        dws_path = os.path.join(path_config.base_path, "src", "basemodels", "dws")
        if dws_path not in sys.path:
            sys.path.insert(0, dws_path)

        config_path = os.path.join(path_config.base_path, "src", "basemodels", "configs", "dws.yaml")
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)

        first_batch = next(iter(testloader))
        del testloader

        if len(first_batch) in [6, 7]:
            params = first_batch[1]
        elif len(first_batch) in [3, 4]:
            params = first_batch[0]
        else:
            raise ValueError("Unexpected batch format for DWS victim model.")
        weights, biases = params
        weight_shapes = tuple(w.permute(0, 1, 3, 2).shape[2:] for w in weights)
        bias_shapes = tuple([b.shape[-1]] for b in biases)

        model = DWSModelForClassification(
            weight_shapes=weight_shapes, bias_shapes=bias_shapes,
            input_features=1, hidden_dim=conf['dim_hidden'], n_hidden=conf['n_hidden'],
            n_classes=10, reduction="max", n_fc_layers=conf['n_fc_layers'],
            set_layer="sab", n_out_fc=conf['n_out_fc'],
            dropout_rate=conf['do_rate'], bn=conf['add_bn'], diagonal=conf['diagonal'],
        ).to(device)
        
        basemodel = ckpt_path['basemodel']
        if load_state and os.path.exists(basemodel):
            state_dict = torch.load(basemodel, weights_only=False, map_location=device)
            if "at_config" in state_dict: # at checkpoint
                state_dict = state_dict['basemodel']
                # state_dict = {k.replace("model.", ""): v for k, v in state_dict['victim_wrapped'].items()}
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded DWS checkpoint from {basemodel}")
        else:
            print(f"Warning: DWS checkpoint not found at {basemodel}")
            model.train()
        model_dict = {"dws": model, "config": {}, "weight_shapes": weight_shapes, "bias_shapes": bias_shapes}
        return VictimWrapper("dws", model_dict, device=device, attack_space=kwargs['attack_space']) if return_wrapped else model_dict

    if victim_model.startswith("inr2vec"):
        from basemodels.inr2vec.models.encoder import Encoder
        from basemodels.inr2vec.models.fc_classifier import FcClassifier

        if inr_config is not None:
            hidden_dim = inr_config.width
        else:
            hidden_dim = 32

        first_batch = next(iter(testloader))
        del testloader

        ws_src = None
        if isinstance(first_batch, (list, tuple)):
            for item in first_batch:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    ws_src = item
                    break

        if ws_src is not None:
            matrix_feat_dim = hidden_dim + 1
        else:
            matrix_feat_dim = 33
        encoder_ckpt_path, classifier_ckpt_path = ckpt_path['basemodel'], ckpt_path['classifier']
        encoder_hidden_dims = [128, 256, 512]
        embedding_dim = kwargs.get("latent_dim", 256)

        if os.path.exists(encoder_ckpt_path):
            ckpt = torch.load(encoder_ckpt_path, weights_only=False, map_location=device)
            if "config" in ckpt:
                from basemodels.training import unwrap_nested_dictconfig
                saved_config = unwrap_nested_dictconfig(ckpt["config"])

                encoder_cfg = saved_config.get("encoder", {})
                encoder_hidden_dims = encoder_cfg.get("hidden_dims", encoder_hidden_dims)
                embedding_dim = encoder_cfg.get("embedding_dim", embedding_dim)

                if dataset_name == 'shapenet':
                    matrix_feat_dim = saved_config['mlp']['hidden_dim']

        encoder = Encoder(input_dim=matrix_feat_dim, 
                          hidden_dims=encoder_hidden_dims, 
                          embed_dim=embedding_dim).to(device)

        if load_state and os.path.exists(encoder_ckpt_path):
            encoder.load_state_dict(ckpt["encoder"])
            print(f"Loaded INR2Vec encoder from {encoder_ckpt_path}")
            encoder.eval()
        else:
            print(f"No INR2Vec encoder from {encoder_ckpt_path}")
            encoder.train()

        classifier_layers = [embedding_dim, 256, 128]
        num_classes = 10

        if os.path.exists(classifier_ckpt_path):
            clf_ckpt = torch.load(classifier_ckpt_path, weights_only=False, map_location=device)
            if "config" in clf_ckpt:
                clf_cfg = clf_ckpt["config"]
                classifier_layers = clf_cfg.get("layers_dim", classifier_layers)
                num_classes = clf_cfg.get("num_classes", num_classes)
            else: # AT checkpoint
                encoder_state = {"encoder": {k.replace("model.", ""):v for k, v in clf_ckpt["basemodel"].items()}}
                encoder.load_state_dict(encoder_state["encoder"])
                # print(clf_ckpt.keys())
                # exit()
                clf_ckpt = {"net": {k.replace("clf.", ""):v for k, v in clf_ckpt["classifier"].items()}}

        classifier = FcClassifier(classifier_layers, num_classes).to(device)

        if load_state and os.path.exists(classifier_ckpt_path) and not load_state_only_encoder:
            # clf_ckpt = torch.load(classifier_ckpt_path, weights_only=False, map_location=device)
            classifier.load_state_dict(clf_ckpt["net"])
            classifier.eval()
            print(f"Loaded INR2Vec classifier from {classifier_ckpt_path}")
        else:
            print(f"No INR2Vec classifier checkpoint found {classifier_ckpt_path}")
            classifier.train()

        model_dict = {"encoder": encoder, "classifier": classifier, "hidden_dim": hidden_dim, "is_3d": dataset_name == 'shapenet'}
        return VictimWrapper("inr2vec", model_dict, device=device, attack_space=kwargs['attack_space']) if return_wrapped else model_dict

    raise ValueError(f"Unsupported victim_model: {victim_model}")


# ---------------------------------------------------------------------------
# Helper: load victim model for adversarial training (standalone function)
# ---------------------------------------------------------------------------

# def load_victim_for_at(
#     victim_model: str,
#     dataset_name: str,
#     base_path: str,
#     device: str,
#     inr_config,
#     path_config: PathConfig,
#     train_loader: DataLoader,
# ) -> Dict[str, Any]:
#     """
#     Helper function to load victim models for adversarial training.

#     Args:
#         victim_model: Model type ('nft', 'nfn', 'dws', 'scalegmn', 'inr2vec')
#         dataset_name: Dataset name
#         base_path: Base path for data
#         device: Device to use
#         inr_config: Optional INR configuration
#         path_config: PathConfig instance
#         train_loader: Training DataLoader (used to get network spec)

#     Returns:
#         Dict containing loaded model components
#     """
#     first_batch = next(iter(train_loader))
#     ws_src = first_batch[0]  # params
#     spec = network_spec_from_wsfeat(WeightSpaceFeatures(*ws_src).to("cpu"), set_all_dims=True)

#     inr_hidden_layers = inr_config.depth if inr_config else 1
#     inr_hidden_features = inr_config.width if inr_config else 32
#     inr_first_omega_0 = inr_config.scale if inr_config else 30.0

#     if victim_model == "nft":
#         config_path = os.path.join(path_config.base_path, "src", "basemodels", "configs", f"inr2array_{dataset_name}.yaml")
#         with open(config_path, 'r') as f:
#             cfg = yaml.safe_load(f)

#         cfg['inr_config'] = {
#             'hidden_layers': inr_hidden_layers,
#             'hidden_features': inr_hidden_features,
#             'first_omega_0': inr_first_omega_0,
#             'hidden_omega_0': inr_first_omega_0,
#         }

#         model_cfg = dict(cfg.get('model', {})) if cfg.get('model') else {}
#         pool_cfg_raw = model_cfg.pop('pool_cls', None) or cfg.get('pool_cls', None)
#         pool_cfg = dict(pool_cfg_raw) if pool_cfg_raw and hasattr(pool_cfg_raw, 'items') else {}
#         pool_cfg.pop('_target_', None)
#         pool_cfg.pop('_partial_', None)
#         model_cfg.pop('_target_', None)

#         pool_cfg['latent_dim'] = model_cfg.get('latent_dim', 256)
#         model_cfg['siren_config'] = cfg['inr_config']
#         nfnet = AutoEncoder(model_cfg, spec, dset_data_type=dataset_name,
#                            pool_cls=partial(PerceiverPooling, **pool_cfg) if pool_cfg else None).to(device)
#         latent_dim = nfnet.latent_shape[-1]
#         clf = TransformerClassifier(latent_dim=latent_dim).to(device)

#         nfnet_ckpt = path_config.get_inr2array_checkpoint(model="nft", dataset=dataset_name, inr_config=inr_config)
#         if os.path.exists(nfnet_ckpt):
#             nfnet.load_state_dict(torch.load(nfnet_ckpt, weights_only=False))
#             print(f"Loaded NFT inr2array checkpoint from {nfnet_ckpt}")
#         clf_ckpt = path_config.get_classifier_checkpoint(model="nft", dataset=dataset_name, inr_config=inr_config)
#         if os.path.exists(clf_ckpt):
#             clf.load_state_dict(torch.load(clf_ckpt, weights_only=False)['clf'])
#             print(f"Loaded NFT classifier checkpoint from {clf_ckpt}")
#         return {"nfnet": nfnet, "clf": clf}

#     elif victim_model == "nfn":
#         from basemodels.utils.nfn.models import MlpHead
#         from basemodels.utils.nfn.nfn.layers import GaussianFourierFeatureTransform, IOSinusoidalEncoding

#         config_path = os.path.join(SRC_DIR, "basemodels", "configs", "nfn.yaml")
#         cfg = load_config(config_path)
#         cfg.hidden_features = inr_hidden_features
#         cfg.hidden_layers = inr_hidden_layers
#         cfg.first_omega_0 = inr_first_omega_0

#         nfnet_cfg = cfg.nfnet if hasattr(cfg, 'nfnet') else cfg.get('nfnet', {})
#         if hasattr(nfnet_cfg, 'to_dict'):
#             nfnet_cfg = nfnet_cfg.to_dict()
#         elif hasattr(nfnet_cfg, '_data'):
#             nfnet_cfg = dict(nfnet_cfg._data)

#         hchannels = nfnet_cfg.get('hchannels', [512, 512, 512])
#         mode = nfnet_cfg.get('mode', 'NP')

#         head_cfg = nfnet_cfg.get('head', {})
#         head_cls = partial(MlpHead, num_out=head_cfg.get('num_out', 10), dropout=head_cfg.get('dropout', 0.5),
#                            lnorm=head_cfg.get('lnorm', False), sigmoid=head_cfg.get('sigmoid', False), pool_mode=mode)

#         inp_enc_cfg = nfnet_cfg.get('inp_enc', {})
#         inp_enc_cls = partial(GaussianFourierFeatureTransform,
#                               mapping_size=inp_enc_cfg.get('mapping_size', 128),
#                               scale=inp_enc_cfg.get('scale', 10)) if inp_enc_cfg else None

#         pos_enc_cfg = nfnet_cfg.get('pos_enc', {})
#         pos_enc_cls = partial(IOSinusoidalEncoding,
#                               max_freq=pos_enc_cfg.get('max_freq', 10),
#                               num_bands=pos_enc_cfg.get('num_bands', 6)) if pos_enc_cfg else None

#         nfnet = InvariantNFN(
#             network_spec=spec, hchannels=hchannels, head_cls=head_cls, mode=mode,
#             feature_dropout=nfnet_cfg.get('feature_dropout', 0.0),
#             normalize=nfnet_cfg.get('normalize', False),
#             lnorm=nfnet_cfg.get('lnorm', None),
#             append_stats=nfnet_cfg.get('append_stats', False),
#             inp_enc_cls=inp_enc_cls, pos_enc_cls=pos_enc_cls,
#         ).to(device)

#         nfn_ckpt = path_config.get_classifier_checkpoint(model="nfn", dataset=dataset_name, inr_config=inr_config)
#         if os.path.exists(nfn_ckpt):
#             nfnet.load_state_dict(torch.load(nfn_ckpt, weights_only=False))
#             print(f"Loaded NFN checkpoint from {nfn_ckpt}")
#         return {"nfnet": nfnet}

#     elif victim_model == "dws":
#         from basemodels.utils.dws.nn.layers import CannibalLayer

#         weight_shapes = [(spec.weight_spec[i].in_features, spec.weight_spec[i].out_features)
#                         for i in range(len(spec.weight_spec))]
#         bias_shapes = [(spec.bias_spec[i].out_features,) for i in range(len(spec.bias_spec))]

#         dws_model = DWSModelForClassification(
#             weight_shapes=weight_shapes, bias_shapes=bias_shapes,
#             hidden_dim=256, n_hidden=3, num_classes=10,
#         ).to(device)

#         dws_ckpt = path_config.get_classifier_checkpoint(model="dws", dataset=dataset_name, inr_config=inr_config)
#         if os.path.exists(dws_ckpt):
#             ckpt = torch.load(dws_ckpt, weights_only=False)
#             if "at_config" in ckpt: # AT 
#                 ckpt = ckpt['basemodel']
#             dws_model.load_state_dict(ckpt)
#             print(f"Loaded DWS checkpoint from {dws_ckpt}")
#         return {"dws": dws_model, "weight_shapes": weight_shapes, "bias_shapes": bias_shapes}

#     elif victim_model == "scalegmn":
#         from basemodels.utils.scalegmn.scalegmn.models import ScaleGMN

#         config_path = os.path.join(SRC_DIR, "basemodels", "configs", f"scalegmn_{dataset_name}.yaml")
#         with open(config_path, 'r') as f:
#             cfg = yaml.safe_load(f)

#         scalegmn_args = cfg['scalegmn_args']
#         scalegmn_args['layer_layout'] = cfg['data']['layer_layout']

#         scalegmn = ScaleGMN(scalegmn_args).to(device)

#         scalegmn_ckpt = path_config.get_classifier_checkpoint(model="scalegmn", dataset=dataset_name, inr_config=inr_config)
#         if os.path.exists(scalegmn_ckpt):
#             scalegmn.load_state_dict(torch.load(scalegmn_ckpt, weights_only=False))
#             print(f"Loaded ScaleGMN checkpoint from {scalegmn_ckpt}")
#         return {"scalegmn": scalegmn, "config": cfg}

#     elif victim_model == "inr2vec":
#         from basemodels.inr2vec.models.encoder import Encoder
#         from basemodels.inr2vec.models.fc_classifier import FcClassifier

#         latent_dim = 256

#         base_model_dir = path_config.get_basemodel_dir(
#             model="inr2vec", dataset=dataset_name, inr_config=inr_config, latent_dim=latent_dim)
#         encoder_ckpt = os.path.join(base_model_dir, "encoder", "ckpts", "best.pt")
#         classifier_ckpt = os.path.join(base_model_dir, "classifier", "ckpts", "best.pt")

#         if os.path.exists(encoder_ckpt):
#             ckpt = torch.load(encoder_ckpt, weights_only=False)
#             saved_config = ckpt.get("config", {})
#             encoder_cfg = saved_config.get("encoder", {})
#             encoder_state = ckpt["encoder"]
#             first_layer_weight = encoder_state.get("layers.0.weight")
#             if first_layer_weight is not None:
#                 input_dim = first_layer_weight.shape[1]
#             else:
#                 input_dim = inr_hidden_features + 1

#             encoder = Encoder(input_dim, encoder_cfg.get("hidden_dims", [128, 256, 512]),
#                             encoder_cfg.get("embedding_dim", latent_dim)).to(device)
#             encoder.load_state_dict(ckpt["encoder"])
#             print(f"Loaded INR2Vec encoder from {encoder_ckpt} (input_dim={input_dim})")
#         else:
#             raise FileNotFoundError(f"INR2Vec encoder checkpoint not found at {encoder_ckpt}")

#         if os.path.exists(classifier_ckpt):
#             clf_ckpt = torch.load(classifier_ckpt, weights_only=False)
#             clf_config = clf_ckpt.get("config", {})
#             layers_dim = clf_config.get("layers_dim", [latent_dim, 256, 128])
#             if layers_dim[0] != latent_dim:
#                 layers_dim = [latent_dim] + layers_dim[1:]
#             num_classes = clf_config.get("num_classes", 10)
#             clf = FcClassifier(layers_dim, num_classes).to(device)
#             clf.load_state_dict(clf_ckpt["net"])
#             print(f"Loaded INR2Vec classifier from {classifier_ckpt}")
#         else:
#             raise FileNotFoundError(f"INR2Vec classifier checkpoint not found at {classifier_ckpt}")

#         mlp_hidden_dim = saved_config.get("mlp_hidden_dim", inr_hidden_features)
#         return {"encoder": encoder, "clf": clf, "hidden_dim": latent_dim, "mlp_hidden_dim": mlp_hidden_dim}

#     else:
#         raise ValueError(f"Unsupported victim_model for AT: {victim_model}")



# def load_at_defended_victim_model(self, victim_wrapper: 'VictimWrapper') -> 'VictimWrapper':
#     if not self.defense_enabled:
#         return victim_wrapper

#     if self.defense_type not in ("embedding_at", "inr_at", "adversarial"):
#         return victim_wrapper

#     # Determine AT checkpoint path
#     at_type = 'embedding' if self.defense_type in ('adversarial', 'embedding_at') else 'inr'
#     defense_checkpoint = self.defense_params.get("checkpoint")

#     if defense_checkpoint is None:
#         defense_checkpoint = os.path.join(
#             self.path_config.get_basemodel_dir(
#                 model=self.victim_model, dataset=self.dataset_name, inr_config=self.inr_config, at_type=at_type
#             ),
#             "best.pt"
#         )

#     if not os.path.exists(defense_checkpoint):
#         print(f"Warning: AT defense checkpoint not found: {defense_checkpoint}")
#         print("Running with base model (no AT defense applied)")
#         return victim_wrapper

#     print(f"Loading adversarial training defense from: {defense_checkpoint}")
#     ckpt = torch.load(defense_checkpoint, weights_only=False, map_location=self.device)

#     # The checkpoint contains 'victim_wrapper' state dict - load directly into VictimWrapper
#     if 'victim_wrapper' in ckpt:
#         victim_wrapper.load_state_dict(ckpt['victim_wrapper'])
#         print(f"  Loaded AT victim wrapper weights (type={at_type})")
#     else:
#         # Legacy format: try old keys for backward compatibility
#         if at_type == 'embedding':
#             if 'encoder' in ckpt and ckpt['encoder'] is not None:
#                 if hasattr(victim_wrapper, 'nfnet') and victim_wrapper.nfnet is not None:
#                     victim_wrapper.nfnet.load_state_dict(ckpt['encoder'])
#                     print("  Loaded AT encoder weights -> nfnet")
#                 elif hasattr(victim_wrapper, 'encoder') and victim_wrapper.encoder is not None:
#                     victim_wrapper.encoder.load_state_dict(ckpt['encoder'])
#                     print("  Loaded AT encoder weights -> encoder")
#             if 'classifier' in ckpt and ckpt['classifier'] is not None:
#                 if hasattr(victim_wrapper, 'clf') and victim_wrapper.clf is not None:
#                     victim_wrapper.clf.load_state_dict(ckpt['classifier'])
#                     print("  Loaded AT classifier weights -> clf")
#         print(f"  Loaded AT weights (legacy format, type={at_type})")

#     print(f"AT defense loaded successfully (type={at_type})")
#     return victim_wrapper