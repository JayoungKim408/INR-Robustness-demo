import torch
import torch.nn as nn
from einops import rearrange
from .util import get_mgrid, sample, compute_psnr, preview_img
from .wt import WT


def build_classifier(conf, device):
    D = conf['siren_dim'][1]
    if conf['model'] == 'WT':
        classifier = WT(dim=D, conf=conf).to(device)
    elif conf['model'] == 'NFN_NP':
        from .nfn_network import InvariantNFN
        classifier = InvariantNFN(dim=D, conf=conf, mode='NP').to(device)
    elif conf['model'] == 'NFN_HNP':
        from .nfn_network import InvariantNFN
        classifier = InvariantNFN(dim=D, conf=conf, mode='HNP').to(device)
    else:
        raise ValueError('unknown model')
    return classifier


def forward(conf,
            image,
            preview_dict,
            build_graph,
            device,
            PI,
            siren_rgb,
            classifier,
            meta_sgd,
            eval_mode=False,
            ):

    # print('TAKING NUM STEPS', inner_steps)

    if conf['is_3d']:
        udf_points, img_flat = image
        N = udf_points.shape[0]
    else:
        N, _, H, W = image.shape
        img_flat = rearrange(image, 'n c h w -> n (h w) c') # [N, H*W, C]
        coords_flat = get_mgrid(H, W, device) # [H*W, 2]
        coords_flat = coords_flat.unsqueeze(0).expand(N, -1, -1) # [N, H*W, 2]

    weight_offset = nn.Parameter(torch.zeros(size=(N, PI), device=device), requires_grad=True)
    for j in range(conf['inner_steps'] + 1):
        last = (j == conf['inner_steps'])

        if conf['is_3d']: # # [N, ST, P_UDF, XYZ=3], [N, ST, P_UDF]
            use_coords = udf_points[:, j] # [N, P_UDF, 3]
            use_image = img_flat[:, j] # [N, P_UDF, 1]
        else:
            use_coords, use_image, indices = sample(conf['sample_size'], coords_flat, img_flat) # [N, 10000, 2], [N, 10000, 3]
            if (j == 0):
                preview_img(siren_rgb, coords_flat, weight_offset, indices, preview_dict, device, H, W)

        pred_rgbw = siren_rgb(use_coords, weight_offset=weight_offset) # [N, 10000, 3]
        assert pred_rgbw.shape[2] == conf['color_channels'], 'pred_rgbw should have C channels'
        loss = ((pred_rgbw - use_image) ** 2).mean(2) # [N, H*W]
        loss_summed = loss.mean(dim=1) # [N]

        if not last:
            grads_wrt_mods = torch.autograd.grad(loss_summed.sum(), [ weight_offset ], create_graph=build_graph)[0] # [N, TI]
            if build_graph:
                assert grads_wrt_mods.requires_grad, 'grads should require grad'

            # update weight offsets
            weight_offset = weight_offset - meta_sgd(j) * grads_wrt_mods # [N, WC]

            if not conf['is_3d']:
                preview_img(siren_rgb, coords_flat, weight_offset, indices, preview_dict, device, H, W)

        else:
            if not eval_mode:
                # compute the PSNR for the last step
                if conf['is_3d']:
                    psnr = torch.zeros(N, device=device)
                else:
                    psnr = compute_psnr(use_image, pred_rgbw)
            else: psnr = None
    # loop over each output token (the amount of incoming values is constant except for first layer)
    vls = siren_rgb.sirens
    shapes = [ p.shape for p in vls ]
    l = []
    c = 0
    for shape in shapes:
        w = weight_offset[:, c:(c+shape.numel())]
        w = w.view(*([N] + list(shape)))
        if conf['cls_loss_weight'] == 0:
            w = w.detach() # speed up, no need to backprop is this is the case
        l.append(w)
        c += shape.numel()
    assert c == siren_rgb.num_params(), 'c should be equal to weight count but %d != %d' % (c, siren_rgb.num_params())
    # make pairs of 2, and skip first layer
    l = [ (l[i], l[i+1]) for i in range(2, len(l) - 1, 2) ]
    logits = classifier(l) # [N, 10]
    return logits, loss_summed.mean(), psnr
