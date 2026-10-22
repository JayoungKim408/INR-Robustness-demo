"""
Utilities for converting INR weights to ScaleGMN graph format.

ScaleGMN uses a graph representation where:
- Nodes: Each neuron is a node with bias as node feature
- Edges: Each connection between neurons is an edge with weight as edge feature
- The graph structure follows the neural network layer topology
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
import torch_geometric
from torch_geometric.data import Data, Batch

from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures


def get_scalegmn_batch_params(config: Dict) -> Dict:

    scalegmn_args = config.get("scalegmn_args", {})
    gnn_args = scalegmn_args.get("gnn_args", {})

    # Check if mask_hidden is needed (based on helpers.py logic)
    equiv_on_hidden = (
        gnn_args.get("msg_equiv_on_hidden", False) or
        gnn_args.get("upd_equiv_on_hidden", False) or
        gnn_args.get("layer_msg_equiv_on_hidden", False) or
        gnn_args.get("layer_upd_equiv_on_hidden", False)
    )

    # Check if first layer mask is needed
    get_first_layer_mask = (
        gnn_args.get("msg_num_mlps", 0) == 3 or
        gnn_args.get("upd_num_mlps", 0) == 3
    )

    return {
        "direction": scalegmn_args.get("direction", "forward"),
        "node_pos_embed": scalegmn_args.get("node_pos_embed", False),
        "edge_pos_embed": scalegmn_args.get("edge_pos_embed", False),
        "equiv_on_hidden": equiv_on_hidden,
        "include_first_layer_mask": get_first_layer_mask,
    }


def get_layer_layout_from_wsfeat(wsfeat: WeightSpaceFeatures) -> List[int]:
    """
    Infer the layer layout (nodes per layer) from WeightSpaceFeatures.

    Args:
        wsfeat: WeightSpaceFeatures with weights tuple of shape:
            - 4D: (batch, channels, out_dim, in_dim) - PyTorch Linear format
            - 3D: (batch, out_dim, in_dim) - legacy format

    Returns:
        List of integers representing nodes per layer [input_dim, hidden1_dim, ..., output_dim]
    """
    weights = wsfeat.weights
    # Handle both 4D (batch, channels, out_dim, in_dim) and 3D (batch, out_dim, in_dim) formats
    # PyTorch Linear stores weights as (out_features, in_features)
    if weights[0].dim() == 4:
        # 4D format: (batch, channels, out_dim, in_dim)
        layout = [weights[0].shape[3]]  # in_dim from first layer
        for w in weights:
            layout.append(w.shape[2])  # out_dim
    else:
        # 3D format: (batch, out_dim, in_dim)
        layout = [weights[0].shape[2]]  # in_dim
        for w in weights:
            layout.append(w.shape[1])  # out_dim
    return layout


def nn_to_edge_index(layer_layout: List[int], device, dtype=torch.long) -> torch.Tensor:
    """
    Create edge index tensor for a fully-connected feedforward network.

    Args:
        layer_layout: List of nodes per layer [input_dim, hidden1_dim, ..., output_dim]
        device: Device to create tensor on
        dtype: Data type for edge index

    Returns:
        Edge index tensor of shape (2, num_edges)
    """
    edge_index = []

    node_offset = 0
    nodes_per_layer = []
    for n in layer_layout:
        nodes_per_layer.append(list(range(node_offset, node_offset + n)))
        node_offset += n

    for i in range(1, len(layer_layout)):
        for j in nodes_per_layer[i - 1]:
            for k in nodes_per_layer[i]:
                edge_index.append([j, k])

    return torch.tensor(edge_index, device=device, dtype=dtype).T


def get_node_types(layer_layout: List[int]) -> torch.Tensor:
    """
    Create node type tensor for position embeddings.

    Input nodes get unique types, hidden nodes share a type per layer,
    output nodes get unique types.
    """
    node_types = []
    type_idx = 0
    for i, n_nodes in enumerate(layer_layout):
        if i == 0:  # input layer - unique type per node
            for _ in range(n_nodes):
                node_types.append(type_idx)
                type_idx += 1
        elif i < len(layer_layout) - 1:  # hidden layers - shared type
            for _ in range(n_nodes):
                node_types.append(type_idx)
            type_idx += 1
        else:  # output layer - unique type per node
            for _ in range(n_nodes):
                node_types.append(type_idx)
                type_idx += 1
    return torch.tensor(node_types)


def get_edge_types(layer_layout: List[int]) -> torch.Tensor:
    """
    Create edge type tensor for position embeddings.

    Edges from input nodes get unique types per input node,
    edges between hidden layers share a type,
    edges to output nodes get unique types per output node.
    """
    edge_types = []
    type_idx = 0
    for i, n_nodes in enumerate(layer_layout[:-1]):
        if i == 0:  # first layer - unique type per input node
            for _ in range(n_nodes):
                for _ in range(layer_layout[i + 1]):
                    edge_types.append(type_idx)
                type_idx += 1
        elif i < len(layer_layout) - 2:  # hidden layers - shared type
            for _ in range(n_nodes):
                for _ in range(layer_layout[i + 1]):
                    edge_types.append(type_idx)
            type_idx += 1
        else:  # last hidden to output - unique type per output node
            for _ in range(layer_layout[i + 1]):
                for _ in range(n_nodes):
                    edge_types.append(type_idx)
                type_idx += 1
    return torch.tensor(edge_types)


def get_hidden_mask(layer_layout: List[int]) -> torch.Tensor:
    """
    Create boolean mask for hidden layer nodes.

    Returns:
        Tensor of shape (num_nodes, 1) with True for hidden nodes.
    """
    hidden_mask = (
        [False] * layer_layout[0] +  # input nodes
        [True] * sum(layer_layout[1:-1]) +  # hidden nodes
        [False] * layer_layout[-1]  # output nodes
    )
    return torch.tensor(hidden_mask).unsqueeze(-1)


def get_first_layer_mask(layer_layout: List[int]) -> torch.Tensor:
    """
    Create boolean mask for first layer (input) nodes.

    Returns:
        Tensor of shape (num_nodes, 1) with True for input nodes.
    """
    first_layer_mask = (
        [True] * layer_layout[0] +  # input nodes
        [False] * sum(layer_layout[1:])  # hidden + output nodes
    )
    return torch.tensor(first_layer_mask).unsqueeze(-1)


def wsfeat_to_graphs(
    wsfeat: WeightSpaceFeatures,
    direction: str = 'forward',
    node_pos_embed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert WeightSpaceFeatures to graph representation (node features, edge features, edge index).

    Args:
        wsfeat: WeightSpaceFeatures with:
            - weights: tuple of tensors with shape:
                - 4D: (batch, channels, in_dim, out_dim) - standard format
                - 3D: (batch, in_dim, out_dim) - legacy format
            - biases: tuple of tensors with shape:
                - 3D: (batch, channels, out_dim) - standard format
                - 2D: (batch, out_dim) - legacy format
        direction: 'forward', 'backward', or 'bidirectional'
        node_pos_embed: Whether to return node type information

    Returns:
        Tuple of (node_features, edge_features, edge_index, node2type)
        - node_features: (batch, num_nodes, feat_dim)
        - edge_features: (batch, num_edges, feat_dim)
        - edge_index: (2, num_edges)
        - node2type: (num_nodes,) or None
    """
    weights = wsfeat.weights
    biases = wsfeat.biases

    device = weights[0].device
    batch_size = weights[0].shape[0]

    # Detect format: 4D (batch, channels, in_dim, out_dim) vs 3D (batch, in_dim, out_dim)
    is_4d_format = weights[0].dim() == 4

    # Infer layer layout
    layer_layout = get_layer_layout_from_wsfeat(wsfeat)
    num_nodes = sum(layer_layout)

    # Create edge index
    edge_index = nn_to_edge_index(layer_layout, device)

    # ScaleGMN expects features with dimension 1 (scalar biases/weights)
    # The channels dimension in 4D format is squeezed out
    feat_dim = 1

    # Build node features from biases
    # Input nodes have no bias, set to 1
    node_features = torch.zeros(batch_size, num_nodes, feat_dim, device=device)

    # Set input node features to 1
    input_dim = layer_layout[0]
    node_features[:, :input_dim, :] = 1.0

    # Fill in bias values for hidden and output nodes
    node_offset = input_dim
    for b in biases:
        # Handle both 3D (batch, channels, out_dim) and 2D (batch, out_dim) formats
        if b.dim() == 3:
            # 3D format: (batch, channels, out_dim) - squeeze channels
            out_dim = b.shape[2]
            node_features[:, node_offset:node_offset + out_dim, 0] = b[:, 0, :]
        else:
            # 2D format: (batch, out_dim)
            out_dim = b.shape[1]
            node_features[:, node_offset:node_offset + out_dim, 0] = b
        node_offset += out_dim

    # Build dense edge feature matrix
    # edge_feat_matrix[batch, src_node, dst_node, feat] stores the edge weight from src to dst
    edge_feat_matrix = torch.zeros(batch_size, num_nodes, num_nodes, feat_dim, device=device)

    row_offset = 0
    col_offset = layer_layout[0]  # Start columns after input nodes
    for w in weights:
        if is_4d_format:
            # 4D format: (batch, channels, out_dim, in_dim) - PyTorch Linear format
            # Need to transpose to (batch, in_dim, out_dim) for edge matrix
            out_dim, in_dim = w.shape[2], w.shape[3]
            # Squeeze channels dimension and transpose to (batch, in_dim, out_dim)
            edge_feat_matrix[:, row_offset:row_offset + in_dim, col_offset:col_offset + out_dim, 0] = w[:, 0, :, :].transpose(1, 2)
        else:
            # 3D format: (batch, out_dim, in_dim)
            out_dim, in_dim = w.shape[1], w.shape[2]
            edge_feat_matrix[:, row_offset:row_offset + in_dim, col_offset:col_offset + out_dim, 0] = w.transpose(1, 2)
        row_offset += in_dim
        col_offset += out_dim

    # Handle direction
    if direction == 'backward':
        edge_feat_matrix = edge_feat_matrix.transpose(-2, -3)
        edge_index = torch.flip(edge_index, [0])

    # Extract edge features using edge index
    # edge_index shape: (2, num_edges)
    # edge_features shape: (batch, num_edges, feat_dim)
    edge_features = edge_feat_matrix[:, edge_index[0], edge_index[1], :]

    # Get node types if requested
    node2type = get_node_types(layer_layout) if node_pos_embed else None

    return node_features, edge_features, edge_index, node2type


def wsfeat_to_pyg_batch(
    wsfeat: WeightSpaceFeatures,
    labels: Optional[torch.Tensor] = None,
    direction: str = 'forward',
    node_pos_embed: bool = False,
    edge_pos_embed: bool = False,
    equiv_on_hidden: bool = True,
    include_first_layer_mask: bool = True,
) -> Batch:

    node_features, edge_features, edge_index, node2type = wsfeat_to_graphs(
        wsfeat, direction=direction, node_pos_embed=node_pos_embed
    )

    batch_size = node_features.shape[0]
    device = node_features.device

    # Get layer layout for masks
    layer_layout = get_layer_layout_from_wsfeat(wsfeat)

    # Compute masks and type tensors
    edge2type = get_edge_types(layer_layout) if edge_pos_embed else None
    mask_hidden = get_hidden_mask(layer_layout) if equiv_on_hidden else None
    mask_first_layer = get_first_layer_mask(layer_layout) if include_first_layer_mask else None

    # Create list of Data objects
    data_list = []
    for i in range(batch_size):
        data = Data(
            x=node_features[i],              # (num_nodes, feat_dim)
            edge_index=edge_index,           # (2, num_edges)
            edge_attr=edge_features[i],      # (num_edges, feat_dim)
        )
        if node2type is not None:
            data.node2type = node2type.to(device)
        if edge2type is not None:
            data.edge2type = edge2type.to(device)
        if mask_hidden is not None:
            data.mask_hidden = mask_hidden.to(device)
        if mask_first_layer is not None:
            data.mask_first_layer = mask_first_layer.to(device)
        if labels is not None:
            data.label = labels[i]
        if direction == 'bidirectional':
            data.bw_edge_index = torch.flip(edge_index, [0])
        data_list.append(data)

    # Batch the data objects
    batch = Batch.from_data_list(data_list)
    return batch


class ScaleGMNINRWrapper(nn.Module):
    """
    Wrapper to make ScaleGMN work with WeightSpaceFeatures input.

    Converts WeightSpaceFeatures -> PyG Batch -> ScaleGMN forward -> logits
    """

    def __init__(
        self,
        scalegmn_model: nn.Module,
        direction: str = 'forward',
        node_pos_embed: bool = False,
        edge_pos_embed: bool = False,
        equiv_on_hidden: bool = True,
        include_first_layer_mask: bool = True,
    ):
        super().__init__()
        self.scalegmn_model = scalegmn_model
        self.direction = direction
        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.equiv_on_hidden = equiv_on_hidden
        self.include_first_layer_mask = include_first_layer_mask

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        """
        Forward pass converting WeightSpaceFeatures to logits.

        Args:
            wsfeat: WeightSpaceFeatures with INR weights/biases

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Convert to PyG batch
        batch = wsfeat_to_pyg_batch(
            wsfeat,
            direction=self.direction,
            node_pos_embed=self.node_pos_embed,
            edge_pos_embed=self.edge_pos_embed,
            equiv_on_hidden=self.equiv_on_hidden,
            include_first_layer_mask=self.include_first_layer_mask,
        )

        # Move batch to same device as model
        device = next(self.scalegmn_model.parameters()).device
        batch = batch.to(device)

        # Forward through ScaleGMN
        logits = self.scalegmn_model(batch)

        return logits
