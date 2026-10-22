import torch
import argparse
import sys
from distutils.util import strtobool


def get_config():
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str)
    args.add_argument('--config', type=str, default=None)

    args.add_argument('--model', type=str, default='WT')
    args.add_argument('--dataset', type=str, default='mnist')
    args.add_argument('--epochs', type=int, default=0)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--auto_cast', type=lambda x: bool(strtobool(x)), default=True)
    args.add_argument('--device', type=str, default='cuda')

    args.add_argument('--dim', type=int, default=128)
    args.add_argument('--siren_depth', type=int, default=4)
    args.add_argument('--omega', type=float, default=30.0)  # SIREN frequency scale (first_omega_0)
    args.add_argument('--save_dir', type=str, default=None)  # Custom save directory for checkpoints

    args.add_argument('--lr_siren_rgb', type=float, default=1e-4)
    args.add_argument('--lr_classifier', type=float, default=1e-4)
    args.add_argument('--lr_sgd_lrs', type=float, default=1e-2)
    args.add_argument('--weight_decay', type=float, default=1e-4)

    args.add_argument('--inner_steps', type=int, default=6)
    args.add_argument('--inner_steps_test', type=int, default=None)
    args.add_argument('--modulation_scale', type=float, default=500)
    args.add_argument('--cls_loss_weight', type=float, default=0.01)
    args.add_argument('--shared_lr', type=lambda x: bool(strtobool(x)), default=False)
    args.add_argument('--classifier_depth', type=int, default=10)
    args.add_argument('--mlp_mult', type=int, default=1)
    args.add_argument('--subsample_points', type=float, default=1.0)
    args.add_argument('--augmentations', type=lambda x: bool(strtobool(x)), default=False)
    args.add_argument('--stages', type=str, default='simultaneous')

    args.add_argument('--victim_model', type=str, default='mwt')
    args.add_argument('--data_name', type=str, default='mnist')
    args.add_argument('--attack_type', type=str, default='transfer2')
    # args.add_argument('--nfn_channels', type=int, default=32)

    conf = vars(args.parse_args())

    if conf['inner_steps_test'] is None:
        conf['inner_steps_test'] = conf['inner_steps']

    c = conf['config']
    if c == 'MWT' or c == None:
        pass # default
    elif c == 'MWT-L':
        conf['dim'] = 256
    elif c == 'MWT-W':
        conf['mlp_mult'] = 4

    elif c == 'WT-sim':
        conf['cls_loss_weight'] = 0.0
        conf['stages'] = 'simultaneous'
    elif c == 'WT-seq':
        conf['cls_loss_weight'] = 0.0
        conf['stages'] = 'sequential'
        
    else:
        raise ValueError(f'Unknown config {c}')
    
    if conf['inner_steps'] != conf['inner_steps_test']:
        assert conf['shared_lr'], 'inner_steps_test different from inner_steps requires shared_lr'

    # convert to a dict
    conf['siren_dim'] = [ conf['dim'] ] * conf['siren_depth']
    conf['nfn_channels'] = [conf['dim']] * conf['classifier_depth']
    conf['command'] = ' '.join(sys.argv)

    print('using config', conf)
    return conf
