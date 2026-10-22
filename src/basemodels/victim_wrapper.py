"""Unified wrapper providing consistent interface across different victim model types."""

from typing import Any, Dict

import torch
from torch import nn

from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from attacks.utils import wsfeat_to_dws
from attacks.scalegmn_utils import wsfeat_to_pyg_batch, get_scalegmn_batch_params


class VictimWrapper(nn.Module):
    def __init__(self, model_type: str, model_dict: Dict[str, Any], device: str = "cuda", attack_space="image"):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.attack_space = attack_space
        self.clf = None
        
        if model_type.startswith("nft"):
            self.model = model_dict["nfnet"]  # AutoEncoder for INR -> embedding
            self.clf = model_dict["clf"]  # TransformerClassifier for embedding -> logits
        elif model_type.startswith("nfn"):
            self.model = model_dict["nfnet"]  # InvariantNFN for INR -> logits
        elif model_type.startswith("dws"):
            self.model = model_dict["dws"]  # DWSModelForClassification
            self.weight_shapes = model_dict.get("weight_shapes")
            self.bias_shapes = model_dict.get("bias_shapes")
        elif model_type.startswith("scalegmn"):
            self.model = model_dict["scalegmn"]
            self.config = model_dict.get("config", {})
        elif model_type.startswith("inr2vec"):
            self.model = model_dict["encoder"]  # INR2Vec Encoder for INR -> embedding
            self.clf = model_dict["classifier"]  # FC classifier for embedding -> logits
            self.mlp_hidden_dim = model_dict.get("hidden_dim", 256)  # Embedding dimension
            self.is_3d = model_dict.get("is_3d", False)
        elif model_type.startswith("mlp"):
            self.model = model_dict["classifier"]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, inp) -> torch.Tensor:
        return self.classify_embedding(inp) if self.attack_space == "embedding" else self.classify_inr(inp)
    
    def classify_inr(self, wsfeat: "WeightSpaceFeatures") -> torch.Tensor:
        
        if self.model_type.startswith("nft"):
            embedding = self.model.encode(wsfeat)
            batch_size = embedding.shape[0]
            if len(embedding.shape) == 3:  # reduce=True
                logits = self.clf(embedding)
            else:
                n_latent = self.model.latent_shape[0] if len(self.model.latent_shape) == 2 else 1
                logits = self.clf(embedding.view(batch_size, n_latent, -1))
            return logits
        # if self.model_type.startswith("nft"):
        #     embedding = self.model.encode(wsfeat)
        #     batch_size = embedding.shape[0]
        #     logits = self.clf(embedding.view(batch_size, 16, -1))
        #     return logits

        elif self.model_type.startswith("nfn"):
            logits = self.model(wsfeat)
            return logits
        
        elif self.model_type.startswith("mlp"):
            logits = self.model(wsfeat)
            return logits

        elif self.model_type.startswith("dws"):
            weights_dws, biases_dws = wsfeat_to_dws(wsfeat)
            logits = self.model((weights_dws, biases_dws))
            return logits

        elif self.model_type.startswith("scalegmn"):
            batch_params = get_scalegmn_batch_params(self.config)
            pyg_batch = wsfeat_to_pyg_batch(wsfeat=wsfeat, **batch_params)
            logits = self.model(pyg_batch)
            return logits

        elif self.model_type.startswith("inr2vec"):
            matrices = self._inr2vec_to_matrices(wsfeat)
            embedding = self.model(matrices)
            logits = self.clf(embedding)
            return logits

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def eval(self):
        super().eval()
        self.model.eval()
        if self.clf is not None:
            self.clf.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        if self.clf is not None:
            self.clf.train(mode)
        return self

    def encode(self, wsfeat: "WeightSpaceFeatures") -> torch.Tensor:
        """
        Encode INR weights to embedding.

        Args:
            wsfeat: WeightSpaceFeatures containing INR weights and biases

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        if self.model_type.startswith("nft"):
            # NFT: INR -> embedding (flat)
            embedding = self.model.encode(wsfeat)
            return embedding.view(embedding.shape[0], -1)  # Flatten to (B, 4096)

        elif self.model_type.startswith("inr2vec"):
            matrices = self._inr2vec_to_matrices(wsfeat)
            embedding = self.model(matrices)
            return embedding

        else:
            raise ValueError(f"encode() not supported for model_type: {self.model_type}")
        
    def _inr2vec_to_matrices(self, wsfeat):
        """Convert wsfeat to matrices for inr2vec encoder."""
        weights = wsfeat.weights
        biases = wsfeat.biases
        batch_size = weights[0].shape[0]
        
        if self.is_3d:
            # PCD inr2vec: get_mlp_params_as_matrix (weight only, no bias concat)
            matrices = []
            for i in range(batch_size):
                middle_weights = []
                for layer_idx in range(1, len(weights) - 1):
                    w = weights[layer_idx][i]
                    if w.dim() == 3:
                        w = w.squeeze(0)
                    middle_weights.append(w)
                if middle_weights:
                    matrix = torch.cat(middle_weights, dim=0)
                else:
                    parts = []
                    for w, b in zip(weights, biases):
                        parts.append(w[i].reshape(-1))
                        parts.append(b[i].reshape(-1))
                    matrix = torch.cat(parts).reshape(-1, self.mlp_hidden_dim)
                matrices.append(matrix)
        else:
            # 2D: weight + bias concat
            from basemodels.inr2vec.image_classification.train_inr2vec import get_mlp_params_as_matrix_from_wsfeat
            matrices = []
            for i in range(batch_size):
                w_tuple = tuple(w[i] for w in weights)
                b_tuple = tuple(b[i] for b in biases)
                matrix = get_mlp_params_as_matrix_from_wsfeat(w_tuple, b_tuple, self.mlp_hidden_dim)
                matrices.append(matrix)
        
        return torch.stack(matrices, dim=0).to(self.device)

    def classify_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify from embedding.

        Args:
            embedding: Flat embedding tensor (batch_size, embedding_dim)

        Returns:
            Logits tensor (batch_size, num_classes)
        """
        if self.model_type.startswith("nft"):
            batch_size = embedding.shape[0]
            if len(embedding.shape) == 3:  # reduce=True
                logits = self.clf(embedding)
            else:
                n_latent = self.model.latent_shape[0] if len(self.model.latent_shape) == 2 else 1
                logits = self.clf(embedding.view(batch_size, n_latent, -1))
            return logits

            # batch_size = embedding.shape[0]
            # n_latent = getattr(self, 'n_latent', 16)
            # return self.clf(embedding.view(batch_size, n_latent, -1))

        elif self.model_type.startswith("inr2vec"):
            return self.clf(embedding)

        else:
            raise ValueError(f"classify_embedding() not supported for model_type: {self.model_type}")

    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Embedding dimension (flat)
        """
        if self.model_type.startswith("nft"):
            return self.model.encoder[-1].latent_dim * getattr(self, 'n_latent', 16)

        elif self.model_type.startswith("inr2vec"):
            return self.model.embed_dim
        
        else:
            return None
