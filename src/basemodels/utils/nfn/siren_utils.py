import math
import numpy as np
import torch
from torch import nn
# from functorch import functional_call, functional_call, functional_call_with_buffers, vmap
from torch.func import vmap, functional_call
from .nfn.common import NetworkSpec


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_mgrid_ffn(height, width):
    """Generates a flattened grid of (x,y) coordinates in [0, 1) range.

    Matches make_ffn.py: np.linspace(0, 1, N+1)[:-1]
    Returns coords in (x, y) order matching make_ffn.py convention.
    """
    y = torch.linspace(0, 1, height + 1)[:-1]
    x = torch.linspace(0, 1, width + 1)[:-1]
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    return coords


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else: # xavier uniform
                self.linear.weight.uniform_( 
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.,
        **args
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output, coords


IMG_shapes = {"mnist": (28, 28, 1), "cifar": (32, 32, 3), "fashion": (28, 28, 1), "shapenet": ()}
SIREN_kwargs = {
    "mnist": {
        "first_omega_0": 30,
        "hidden_features": 32,
        "hidden_layers": 1,
        "hidden_omega_0": 30.0,
        "in_features": 2,
        "out_features": 1,
        "outermost_linear": True,
    },
    "fashion": {
        "first_omega_0": 30,
        "hidden_features": 32,
        "hidden_layers": 1,
        "hidden_omega_0": 30.0,
        "in_features": 2,
        "out_features": 1,
        "outermost_linear": True,
    },
    "cifar": {
        "first_omega_0": 30,
        "hidden_features": 32,
        "hidden_layers": 1,
        "hidden_omega_0": 30.0,
        "in_features": 2,
        "out_features": 3,
        "outermost_linear": True,
    },
    "shapenet": {
        "first_omega_0": 30,
        "hidden_features": 128,
        "hidden_layers": 1,
        "hidden_omega_0": 30.0,
        "in_features": 3,
        "out_features": 1,
        "outermost_linear": True,
    }
}

def get_batch_inr(dset_type, inr_config=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if inr_config:
        siren_kwargs = {
            "first_omega_0": inr_config.scale,
            "hidden_features": inr_config.width,
            "hidden_layers": inr_config.depth,
            "hidden_omega_0": inr_config.scale,
            "in_features": SIREN_kwargs[dset_type]['in_features'],
            "out_features": SIREN_kwargs[dset_type]['out_features'],
            "outermost_linear": SIREN_kwargs[dset_type]['outermost_linear'],
        }
        siren = Siren(**siren_kwargs).to(device)
    else:
        siren = Siren(**SIREN_kwargs[dset_type]).to(device)
    
    param_names = list(dict(siren.named_parameters()).keys())
    initial_params = tuple(p.clone() for p in siren.parameters())
    
    if dset_type == "shapenet":
        coords = get_mgrid(20, dim=3).to(device)  # 32^3 = 32768 points
        
        def func_inp(params_tuple):
            params_dict = dict(zip(param_names, params_tuple))
            values = functional_call(siren, params_dict, (coords,))[0]
            return values.squeeze(-1)

        return vmap(func_inp, in_dims=(0,)), initial_params
    
    mgrid_len = 28 if dset_type in ["mnist", "fashion"] else 32
    coords = get_mgrid(mgrid_len, 2).to(device)
    img_shape = IMG_shapes[dset_type]
    
    def func_inp(params_tuple):
        params_dict = dict(zip(param_names, params_tuple))
        values = functional_call(siren, params_dict, (coords,))[0]
        return torch.permute(values.reshape(*img_shape), (2, 0, 1))
    
    return vmap(func_inp, in_dims=(0,)), initial_params

def get_spatial_batch_inr(dset_type, inr_config=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if inr_config:
        siren_kwargs = {
            "first_omega_0": inr_config.scale,
            "hidden_features": inr_config.width,
            "hidden_layers": inr_config.depth,
            "hidden_omega_0": inr_config.scale,
            "in_features": SIREN_kwargs[dset_type]['in_features'],
            "out_features": SIREN_kwargs[dset_type]['out_features'],
            "outermost_linear": SIREN_kwargs[dset_type]['outermost_linear'],
        }
        siren = Siren(**siren_kwargs).to(device)
    else:
        siren = Siren(**SIREN_kwargs[dset_type]).to(device)
    
    param_names = list(dict(siren.named_parameters()).keys())
    initial_params = tuple(p.clone() for p in siren.parameters())
    
    if dset_type == "shapenet":
        coords = get_mgrid(20, dim=3).to(device)
        
        def siren_at_coord(params_tuple, coord):
            params_dict = dict(zip(param_names, params_tuple))
            return functional_call(siren, params_dict, (coord[None],))[0].squeeze(0)
        
        batch_siren_at_coord = vmap(siren_at_coord, (0, 0))
        
        def spatial_siren(params):
            width = int(round(params[0].shape[0] ** (1/3)))
            def reshape(arr):
                return arr.reshape(width, width, width, *arr.shape[1:])
            params = tuple(reshape(p) for p in params)
            param_idcs = torch.round((params[0].shape[0] - 1) * (coords + 1) / 2).long()
            idxd_params = tuple(
                p[param_idcs[:, 0], param_idcs[:, 1], param_idcs[:, 2]]
                for p in params
            )
            values = batch_siren_at_coord(idxd_params, coords)
            return values.squeeze(-1)
        
        return vmap(spatial_siren, in_dims=(0,)), initial_params
    
    mgrid_len = 28 if dset_type in ["mnist", "fashion"] else 32
    coords = get_mgrid(mgrid_len, 2).to(device)
    img_shape = IMG_shapes[dset_type]
    
    def siren_at_coord(params_tuple, coord):
        params_dict = dict(zip(param_names, params_tuple))
        return functional_call(siren, params_dict, (coord[None],))[0].squeeze(0)
    
    batch_siren_at_coord = vmap(siren_at_coord, (0, 0))
    
    def spatial_siren(params):
        width = int(math.sqrt(params[0].shape[0]))
        def reshape(arr):
            return arr.reshape(width, width, *arr.shape[1:])
        params = tuple(reshape(p) for p in params)
        param_idcs = torch.round((params[0].shape[0] - 1) * (coords + 1) / 2).long()
        idxd_params = tuple(p[param_idcs[:, 0], param_idcs[:, 1]] for p in params)
        values = batch_siren_at_coord(idxd_params, coords)
        return torch.permute(values.reshape(*img_shape), (2, 0, 1))
    
    return vmap(spatial_siren, in_dims=(0,)), initial_params

# def get_spatial_batch_siren(dset_type, siren_config=None, device="cuda"):
#     if siren_config:
#         siren_kwargs = {
#             "first_omega_0": siren_config['first_omega_0'],
#             "hidden_features": siren_config['hidden_features'],
#             "hidden_layers": siren_config['hidden_layers'],
#             "hidden_omega_0": siren_config['hidden_omega_0'],
#             "in_features": SIREN_kwargs[dset_type]['in_features'],
#             "out_features": SIREN_kwargs[dset_type]['out_features'],
#             "outermost_linear": SIREN_kwargs[dset_type]['outermost_linear'],
#         }
#         siren = Siren(**siren_kwargs).to(device)
#     else:
#         siren = Siren(**SIREN_kwargs[dset_type]).to(device)
    
#     param_names = list(dict(siren.named_parameters()).keys())
#     initial_params = tuple(p.clone() for p in siren.parameters())
    
#     mgrid_len = 28 if dset_type in ["mnist", "fashion"] else 32
#     coords = get_mgrid(mgrid_len, 2).to(device)
#     img_shape = IMG_shapes[dset_type]
    
#     def siren_at_coord(params_tuple, coord):
#         params_dict = dict(zip(param_names, params_tuple))
#         return functional_call(siren, params_dict, (coord[None],))[0].squeeze(0)
    
#     batch_siren_at_coord = vmap(siren_at_coord, (0, 0))
    
#     def spatial_siren(params):
#         width = int(math.sqrt(params[0].shape[0]))
#         def reshape(arr):
#             return arr.reshape(width, width, *arr.shape[1:])
#         params = tuple(reshape(p) for p in params)
#         param_idcs = torch.round((params[0].shape[0] - 1) * (coords + 1) / 2).long()
#         idxd_params = tuple(p[param_idcs[:, 0], param_idcs[:, 1]] for p in params)
#         values = batch_siren_at_coord(idxd_params, coords)
#         return torch.permute(values.reshape(*img_shape), (2, 0, 1))
    
#     return vmap(spatial_siren, in_dims=(0,)), initial_params


# def get_spatial_batch_inr(dset_type, inr_config=None, device="cuda"):
#     """Unified function to get spatial batch INR renderer for either SIREN or FFN.

#     Args:
#         dset_type: 'mnist', 'fashion', or 'cifar'
#         inr_type: 'siren' or 'ffn'
#         inr_config: Optional config dict for INR architecture
#         device: Device to place the model on

#     Returns:
#         (batch_render_fn, initial_params) tuple
#     """
#     return get_spatial_batch_siren(dset_type, siren_config=inr_config, device=device)


def unprocess_img_arr(arr):
    arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    arr = np.clip(arr, -1, 1)
    return (255 * (arr + 1) / 2).astype(np.uint8)


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)


class SimpleHyperNetwork(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_size, hidden_size=256, hidden_layers=1):
        super().__init__()
        self.network_spec = network_spec
        layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        num_params = network_spec.get_num_params()
        layers.append(nn.Linear(hidden_size, num_params))
        self.hnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.hnet(x)


class HyperNetwork(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_size, hidden_size=256, hidden_layers=1):
        super().__init__()
        self.network_spec = network_spec
        self.hnets = nn.ModuleList([])
        # construct a hypernetwork for each weight and bias in each layer
        for i in range(len(network_spec)):
            self.hnets.append(self._get_hnet(in_size, network_spec.weight_spec[i].shape, hidden_size, hidden_layers))
            self.hnets.append(self._get_hnet(in_size, network_spec.bias_spec[i].shape, hidden_size, hidden_layers))

    def _get_hnet(self, in_size, out_shape, hidden_size, hidden_layers=1):
        out_size = np.prod(out_shape)
        hnet = [nn.Linear(in_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            hnet.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        hnet.append(nn.Linear(hidden_size, out_size))
        hnet = nn.Sequential(*hnet)
        # Following: https://github.com/vsitzmann/siren/blob/master/meta_modules.py#L135
        if len(out_shape) == 1:  # outputting bias
            hnet[-1].apply(hyper_bias_init)
        else:  # outputting weight
            hnet[-1].apply(lambda m: hyper_weight_init(m, out_shape[-1]))
        return hnet

    def forward(self, x):
        outs = []
        for hnet in self.hnets:
            outs.append(hnet(x))
        return torch.cat(outs, dim=1)