import os
import time
import torch
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader, Subset
from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset, get_SirenOriginalAndEmbeddingDataset
from basemodels.utils.nfn.siren_utils import get_batch_inr
from basemodels.model_loading import load_victim_model
from basemodels.model_loading import load_transfer as run_train_transfer_surrogate

from basemodels.training import *

from utils.wandb_manager import WandbManager
from utils.path_config import PathConfig

from attacks.utils import BatchedINRBuilder, SPLIT_POINTS
from attacks.mwt_e2e_attack import main as run_end_to_end_attack
from attacks.hypernet.train_hypernet.train_img_neural_process import main as run_train_hypernet
from attacks.surrogate.train_surrogate import main as run_train_surrogate

from attacks.attack_methods import (
    run_pgd,
    run_pgd_against_surrogate,
    run_image_gaussian_attack,
    run_inr_perturbation_attack,
    run_inr_pgd,
    run_embedding_pgd,
    run_evaluate_classifiers,
    run_square_attack,
    run_inr_square_attack,
    run_inr_autoattack,
    run_deepfool_against_surrogate,
)
from attacks.malt import (
    run_malt,
    run_malt_against_surrogate, 
)
from attacks.run_acg import run_acg_against_surrogate
from attacks.run_cw import run_cw_against_surrogate
from attacks.run_onepixel import run_onepixel
from attacks.unroll_inr import run_unroll_time_measurement

from defences.adversarial_training import adversarial_training

class Attacks:
    def __init__(self,
                 attack_space: str,
                 attack_type: str,
                 dataset_name: str,
                 victim_model: str,
                 base_path: str,
                 surrogate_type: Optional[str] = None,
                 wandb_manager: Optional[WandbManager] = None,
                 defense_config: Optional[Dict[str, Any]] = None,
        ):

        self.attack_space = attack_space
        self.attack_type = attack_type
        self.dataset_name = dataset_name
        self.victim_model = victim_model
        self.base_path = base_path
        self.surrogate_type = surrogate_type
        self.path_config = PathConfig(base_path=base_path)

        self.wandb = wandb_manager

        # Defense configuration
        self.defense_config = defense_config
        self.defense_enabled = defense_config is not None and defense_config.get("enabled", False)
        self.defense_type = defense_config.get("type") if defense_config else None
        self.defense_params = defense_config.get("params", {}) if defense_config else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate attack name for logging
        self.attack_name = f"{self.attack_space}_{self.attack_type}_{self.surrogate_type}_{self.dataset_name}_{self.victim_model}_{time.strftime('%m%d_%H%M%S')}"
        self.exp_plot_path = self.path_config.get_attack_plot_dir(
            attack_type=self.attack_type,
            surrogate_type=self.surrogate_type,
            dataset=self.dataset_name,
            victim_model=self.victim_model,
        )
        self.inr_config = None
        self.custom_siren_path = None
        self.ddp_info = None

    def get_siren_path(self):
        """Get SIREN path (custom or INR-config-aware default)."""
        if self.custom_siren_path:
            return self.custom_siren_path
        if self.inr_config is not None:
            return self.inr_config.get_inr_path(self.base_path, self.dataset_name)
        return os.path.join(self.base_path, "sirens", f"siren_{self.dataset_name}_wts")
    
    def set_defense_config(self, defense_config: Dict[str, Any]):

        self.defense_config = defense_config
        self.defense_enabled = defense_config.get("enabled", False)
        self.defense_type = defense_config.get("type")
        self.defense_params = defense_config.get("params", {})

        # if self.defense_type and self.defense_type not in self.SUPPORTED_DEFENSES:
        #     raise ValueError(f"Unsupported defense type: {self.defense_type}. "
        #                    f"Choose from {self.SUPPORTED_DEFENSES}")

    def ensure_basemodel_trained(self, **kwargs) -> Optional[Dict[str, str]]:

        ckpt_path = self.path_config.get_basemodel_checkpoint(
            model=self.victim_model,
            dataset=self.dataset_name,
            inr_config=self.inr_config,
            latent_dim=kwargs.get('latent_dim', 256),
            defense_type=self.defense_type if self.defense_enabled else None,
            defense_params=self.defense_params if self.defense_enabled else None
        )

        basemodel_path = ckpt_path['basemodel']
        classifier_ckpt_path = ckpt_path.get('classifier', None)
        if os.path.exists(basemodel_path):
            if classifier_ckpt_path is not None:
                if not os.path.exists(classifier_ckpt_path) and not kwargs.get('resume_training', False):
                    print(f"{self.victim_model.upper()} encoder checkpoint found: {ckpt_path}")
                    print(f"But classifier checkpoint missing: {classifier_ckpt_path}")
                    print(f"Training classifier only...")
                    kwargs['train_encoder'] = False
                    kwargs['train_classifier'] = True
                elif kwargs.get('resume_training', False):
                    if not os.path.exists(classifier_ckpt_path):
                        print(f"Resuming training {self.victim_model.upper()} encoder: {ckpt_path}")
                        print(f"Classifier checkpoint missing: {classifier_ckpt_path}")
                        print(f"Training both encoder and classifier...")
                        kwargs['train_encoder'] = True
                        kwargs['train_classifier'] = True
                    else:
                        print(f"Resuming training {self.victim_model.upper()} encoder: {ckpt_path}")
                else: 
                    print(f"Base model checkpoint found: {ckpt_path}")
                    return ckpt_path
            elif kwargs.get('resume_training', False):
                print(f"Resuming training {self.victim_model.upper()} encoder: {ckpt_path}")
                kwargs['train_encoder'] = True
            else:
                print(f"Base model checkpoint found: {ckpt_path}")
                return ckpt_path
        else:
            print(f"Base model checkpoint NOT found: {ckpt_path}")
        # print(f"Training {self.victim_model.upper()} for {self.dataset_name}...")

        print(f"INR config: depth={self.inr_config.depth} width={self.inr_config.width} scale={self.inr_config.scale}")

        try:
            if self.defense_enabled:
                adv_result = self.train_adversarial(**kwargs)
                result = {"model_type": adv_result["model_type"],
                          "status": adv_result["status"]}
                if self.victim_model in ["nft", "inr2vec"]:
                    result['checkpoint'] = {"encoder": basemodel_path, "classifier": adv_result["checkpoint"]} 
                else: 
                    result['checkpoint'] = adv_result["checkpoint"]
            else:
                result = self.train_basemodel(**kwargs)
            # Check if training succeeded by checking result status and checkpoint existence
            if result.get("status") == "failed":
                print(f"Base model training failed: {result.get('error', 'Unknown error')}")
                return None
            # Double-check checkpoint actually exists after training
            checkpoint = result.get('checkpoint')
            if isinstance(checkpoint, dict):
                if not os.path.exists(checkpoint.get('encoder', '')) or not os.path.exists(checkpoint.get('classifier', '')):
                    print(f"Training completed but one or more checkpoints not found at: {checkpoint}")
                    return None
            elif isinstance(checkpoint, str):
                if not os.path.exists(checkpoint):
                    print(f"Training completed but checkpoint not found at: {checkpoint}")
                    return None
            else:
                print(f"Training completed but unexpected checkpoint format: {checkpoint}")
                return None
            print(f"Base model training completed. Checkpoint saved to: {result.get('checkpoint')}")
            return ckpt_path
        except Exception as e:
            print(f"Failed to train base model: {e}")
            return None

    # def get_diffusion_denoiser(self) -> Optional[DiffusionDenoiser]:
    #     """Get diffusion denoiser if defense is configured.

    #     Returns:
    #         DiffusionDenoiser instance or None
    #     """
    #     if not self.defense_enabled or self.defense_type != "diffusion":
    #         return None

    #     config = DiffusionConfig(
    #         timestep=self.defense_params.get("timestep", 100),
    #         multistep=self.defense_params.get("multistep", False),
    #         ddim_steps=self.defense_params.get("ddim_steps", 10),
    #         mode=self.defense_params.get("mode", "denoise_only"),
    #         model_path=self.defense_params.get("model_path"),
    #     )

    #     return DiffusionDenoiser(
    #         dataset=self.dataset_name,
    #         config=config,
    #         device=str(self.device),
    #     )

    # def _load_surrogate_for_defense(self, testloader: DataLoader) -> tuple:
    #     """
    #     Load surrogate model for diffusion defense with surrogate mode.

    #     Returns:
    #         Tuple of (surrogate, weight_shapes, bias_shapes) or (None, None, None) if not available
    #     """
    #     from src.attacks.surrogate.train_surrogate import I2IMLP, get_inr_param_count_from_shapes
    #     from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures

    #     side = 32 if self.dataset_name == "cifar" else 28
    #     c_out = 3 if self.dataset_name == "cifar" else 1

    #     # Get weight/bias shapes from first batch
    #     first_batch = next(iter(testloader))
    #     # first_batch[1] is the INR weights tuple: (weights_tuple, biases_tuple)
    #     sample_wsfeat = WeightSpaceFeatures(*first_batch[1]).to("cpu")
    #     weight_shapes = [tuple(w.shape[1:]) for w in sample_wsfeat.weights]
    #     bias_shapes = [tuple(b.shape[1:]) for b in sample_wsfeat.biases]

    #     total_params = get_inr_param_count_from_shapes(weight_shapes, bias_shapes)

    #     # Create surrogate model
    #     surrogate = I2IMLP(
    #         in_features=c_out * side * side,
    #         out_features=total_params,
    #         hidden_features=512,
    #         hidden_layers=3,
    #     ).to(self.device)

    #     # Try to load pre-trained surrogate
    #     surrogate_dir = self.path_config.get_surrogate_checkpoint_dir(
    #         attack_type="i2i-naive",
    #         dataset=self.dataset_name,
    #         victim_model=self.victim_model,
    #         inr_params=self.inr_config,
    #     )
    #     surrogate_ckpt = os.path.join(surrogate_dir, "best_surrogate.pt")
    #     if os.path.exists(surrogate_ckpt):
    #         print(f"Loading surrogate for defense from {surrogate_ckpt}")
    #         surrogate.load_state_dict(torch.load(surrogate_ckpt, weights_only=False, map_location=self.device))
    #         surrogate.eval()
    #         return surrogate, weight_shapes, bias_shapes
    #     else:
    #         print(f"Warning: Surrogate not found at {surrogate_ckpt}")
    #         print("Cannot use surrogate defense mode without trained surrogate.")
    #         print("Falling back to denoise_only mode (using pre-computed INR weights).")
    #         return None, None, None


    def get_result_dict(self, result: Any = None) -> Dict[str, Any]:

        if result is None:
            result = {}

        standardized = result
        for key in ["error_msg", "checkpoint_path", "training_time_sec", "attack_time_sec"]:
            if key in result:
                standardized[key] = result[key]

        return standardized

    def train_basemodel(
        self,
        model_type: str = None,
        # gpu_ids: List[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        model_type = model_type or self.victim_model
        is_main = self.ddp_info['rank'] == 0
        if is_main:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} on {self.dataset_name.upper()}")
            print(f"{'='*60}\n")

        training_func_map = {
            "nfn": train_nfn,
            "nft": train_nft,
            "scalegmn": train_scalegmn,
            "mwt": train_mwt,
            "dws": train_dws,
            "inr2vec": train_inr2vec,
            "mlp": train_mlp,
        }

        try:
            result = training_func_map[model_type.split("_")[0]](
                dataset_name=self.dataset_name,
                base_path=self.base_path,
                **kwargs
            )
            if is_main:
                print(f"\n{'='*60}")
                print(f"Training completed for {model_type.upper()}")
            if isinstance(result, dict):
                for key, path in result.items():
                    if is_main: print(f"  {key}: {path}")
            else:
                print(f"  Checkpoint: {result}")
            if is_main: print(f"{'='*60}\n")
            return {"model_type": model_type, "checkpoint": result, "status": "success"}

        except Exception as e:
            if is_main: print(f"Training failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return {"model_type": model_type, "error": str(e), "status": "failed"}
        
    def train_adversarial(
        self,
        model_type: str = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        kwargs['attack_space'] = 'image' if self.defense_type == "inr_at" else 'embedding'

        model_type = model_type or self.victim_model
        is_main = self.ddp_info['rank'] == 0
        if is_main:
            print(f"\n{'='*60}")
            print(f"Training Adversarial {model_type.upper()} on {self.dataset_name.upper()}")
            print(f"{'='*60}\n")

        try:
            result = adversarial_training(
                dataset_name=self.dataset_name,
                device=self.device,
                at_type=self.defense_type,
                params=self.defense_params,
                **kwargs
            )
            if is_main:
                print(f"\n{'='*60}")
                print(f"Training completed for {model_type.upper()}")
            if isinstance(result, dict):
                for key, path in result.items():
                    if is_main: print(f"  {key}: {path}")
            else:
                print(f"  Checkpoint: {result}")
            if is_main: print(f"{'='*60}\n")
            return {"model_type": model_type, "checkpoint": result, "status": "success"}

        except Exception as e:
            if is_main: print(f"Training failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return {"model_type": model_type, "error": str(e), "status": "failed"}


    SURROGATE_TRAIN_FUNC = {
        "i2i-hypernet": run_train_hypernet,
        "i2i-fd": run_train_surrogate,
        "i2i-naive": run_train_surrogate,
        "i2e-fd": run_train_surrogate,
        "i2e-naive": run_train_surrogate,
        "transfer": run_train_transfer_surrogate,
    }
    DATALOADER_MAP = {
        "inr": get_SirenAdnOriginalDataset,
        "embedding": get_SirenOriginalAndEmbeddingDataset,
    }
    # ------------------------------
    # Public API
    # ------------------------------
    def run(
        self,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:

        kwargs["path_config"] = self.path_config
        kwargs["inr_config"] = self.inr_config
        kwargs["exp_plot_path"] = self.exp_plot_path
        kwargs["victim_model"] = self.victim_model
        kwargs["attack_type"] = self.attack_type
        kwargs["surrogate_type"] = self.surrogate_type
        kwargs["attack_space"] = self.attack_space
        kwargs["wandb_manager"] = self.wandb
        kwargs["ddp_info"] = self.ddp_info
        kwargs['defense_type'] = self.defense_type

        # Build dataloaders
        dataloader_kwargs = {"siren_path": self.inr_config.get_inr_path(base_path=self.base_path, dataset=self.dataset_name),
                             "siren_prefix": self.inr_config.prefix,
                             "data_path": self.path_config.data_path, 
                             "split_points": SPLIT_POINTS[self.dataset_name],
                             "num_test_samples": kwargs.get("num_test_samples", 10000),
                             }
        

        def _build_dataloader(loader_type="inr", split=["train", "val", "test"]):
            if kwargs.get("embedding_path", None) and loader_type == "embedding":
                dataloader_kwargs["embedding_path"] = kwargs["embedding_path"]

            if "num_test_samples" in dataloader_kwargs:
                num_test_samples = dataloader_kwargs.pop("num_test_samples")
            else: 
                num_test_samples = None
            datasets = {s: self.DATALOADER_MAP[loader_type](**dataloader_kwargs, split=s) for s in split}
            
            if num_test_samples is not None and len(datasets['test']) > num_test_samples:
                datasets['test'] = Subset(datasets['test'], range(num_test_samples))

            return tuple(DataLoader(dset, batch_size=kwargs.get("batch_size", 128), shuffle=s=='train', num_workers=2, drop_last=False) for s, dset in datasets.items())
        
        is_main = self.ddp_info['rank'] == 0
        basemodel_ckpt_path = self.ensure_basemodel_trained(**kwargs)

        if basemodel_ckpt_path is None:
            raise RuntimeError(
                f"Base model training failed for {self.victim_model} on {self.dataset_name}. "
                f"Cannot proceed with attack."
            )
        
        need_surrogate = (
            self.attack_space == "image"
            and self.attack_type in ["pgd", "deepfool", "malt", "acg", "cw"]
        )

        need_embedding = (
            self.surrogate_type is not None
            and self.attack_space == "image"
            and self.attack_type == "pgd"
            and self.surrogate_type.startswith("i2e")
        ) or kwargs.get("use_precomputed_embs", False)

        if need_surrogate:
            kwargs["exp_ckpt_path"] = self.path_config.get_surrogate_checkpoint_path(
                dataset=self.dataset_name,
                victim_model=self.victim_model,
                surrogate_type=self.surrogate_type,
                inr_params=self.inr_config,
                attack_params=kwargs,
            )

        if need_embedding:
            kwargs["embedding_path"] = self.path_config.get_embedding_path(
                model=self.victim_model,
                dataset=self.dataset_name,
                inr_config=self.inr_config,
                latent_dim=kwargs.get("latent_dim", 256),
            )

        kwargs['dataloader'] = _build_dataloader(loader_type="embedding" if need_embedding else "inr")

        # load victim model and wrap with AT defense if enabled
        victim_wrapped = load_victim_model(
                                        dataset_name=self.dataset_name,
                                        device=self.device, 
                                        load_state=True, 
                                        return_wrapped=True,
                                        ckpt_path=basemodel_ckpt_path,
                                        **kwargs)
                                        
        # Apply AT defense if enabled
        # victim_wrapped = load_at_defended_victim_model(victim_wrapper=victim_wrapped) #TODO
        victim_wrapped.eval()

        kwargs["victim_wrapped"] = victim_wrapped

        # Build INR helper
        sidelen = 32 if self.dataset_name == "cifar" else 28
        c_out = 3 if self.dataset_name == "cifar" else 1
        builder = BatchedINRBuilder(
            sidelen=sidelen,
            hidden_features=self.inr_config.width,
            hidden_layers=self.inr_config.depth,
            first_omega_0=self.inr_config.scale,
            hidden_omega_0=self.inr_config.scale,
            c_out=c_out,
            steps=kwargs.get("inr_fitting_steps", 5000),
            lr=kwargs.get("inr_fitting_lr", 5e-4),
            verbose=kwargs.get("verbose", False),
        )

        kwargs["builder"] = builder
        kwargs["renderer"] = get_batch_inr(self.dataset_name)[0]
        # def renderer_for_attack(wsfeat, if_wsfeat=True, requires_grad=False):
        #     imgs = builder.build_img_from_wts_and_bs(wsfeat, if_wsfeat, requires_grad)
        #     # [0,1] -> [-1,1] 로 되돌리기
        #     return imgs * 2.0 - 1.0

        # kwargs["renderer"] = builder.build_img_from_wts_and_bs
        

        result = {}
        if self.attack_type == "unroll":
            return run_unroll_time_measurement( dataset_name=self.dataset_name, **kwargs )
        
        if kwargs.get("evaluate_classifier", False):
            if is_main: print("Evaluating classifier accuracy on clean test set")
            kwargs["dataloader"] = (None, None, _build_dataloader(loader_type="inr", split=["test"])[0])
            res = run_evaluate_classifiers(**kwargs)
            if is_main: print(f"Classifier accuracy on {self.dataset_name}: {res['cls_acc']:.4f}")
            result = result | res


        if self.attack_space == "image":
            if self.attack_type == "gaussian":
                kwargs['dataloader'] = _build_dataloader(loader_type="inr", split=["test"])
                return run_image_gaussian_attack(self, basemodel_ckpt_path=basemodel_ckpt_path, **kwargs)
            
            if self.attack_type == "e2e":
                assert self.victim_model == "mwt", "e2e attack only supports MWT victim model"
                return run_end_to_end_attack(base_path=self.base_path, **kwargs)

            if self.attack_type == "square_attack":
                return run_square_attack(dataset_name=self.dataset_name, **kwargs)
            
            if self.attack_type == "onepixel":
                return run_onepixel(dataset_name=self.dataset_name, **kwargs)
            
            kwargs['surrogate'] = self.SURROGATE_TRAIN_FUNC[self.surrogate_type](
                dataset_name=self.dataset_name,
                type="FD" if "fd" in self.surrogate_type else "Naive",
                **kwargs,
            )

            if self.attack_type == "cw":
                return run_cw_against_surrogate(dataset_name=self.dataset_name, **kwargs)
                
            if self.attack_type == "acg":
                return run_acg_against_surrogate(dataset_name=self.dataset_name, **kwargs)

            if self.attack_type == "malt":
                if kwargs.get('attack_against_surrogate', False):
                    return run_malt_against_surrogate(dataset_name=self.dataset_name, **kwargs)
                else:
                    return run_malt(dataset_name=self.dataset_name, **kwargs)

            if self.attack_type == "deepfool":
                return run_deepfool_against_surrogate(dataset_name=self.dataset_name, **kwargs)

            if self.attack_type == "pgd":
                if kwargs.get('attack_against_surrogate', False):
                    res = run_pgd_against_surrogate(dataset_name=self.dataset_name, **kwargs)
                else:
                # if kwargs.get('attack_against_victim', True):
                    res = run_pgd(dataset_name=self.dataset_name, **kwargs)
                result = result | res
            
            self.wandb.log(result)
            return result

        elif self.attack_space == "inr":
            
            if self.attack_type == "inr_square_attack":
                return run_inr_square_attack(dataset_name=self.dataset_name, **kwargs)
            
            if self.attack_type == "inr_autoattack":
                return run_inr_autoattack(dataset_name=self.dataset_name, **kwargs)
            
            if self.attack_type in ('gaussian', 'transfer'):
                return run_inr_perturbation_attack(self, basemodel_ckpt_path=basemodel_ckpt_path, **kwargs)
            else : #  self.attack_type in ("i2i-hypernet", "i2i-fd", "i2i-naive", "i2e-fd", "i2e-naive"):
                return run_inr_pgd(dataset_name=self.dataset_name, **kwargs)
            # else:
            #     raise ValueError(f"Unsupported attack_type for inr space: {self.attack_type}")

        # Embedding-space PGD attack on NFN embeddings (NFT pipeline only).
        elif self.attack_space == "embedding":
            if self.surrogate_type in ("i2e-fd", "i2e-naive") and self.victim_model in ("nft", "inr2vec"):
                kwargs['dataloader'] = _build_dataloader(loader_type="embedding")
                return run_embedding_pgd(
                    dataset_name=self.dataset_name,
                    **kwargs)

            else:
                raise ValueError(f"Unsupported attack_type for embedding space: {self.attack_type}")

        else:
            raise ValueError(f"Unsupported attack_space: {self.attack_space}")

