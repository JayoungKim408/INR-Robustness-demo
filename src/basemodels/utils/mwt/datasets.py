
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from .data.augment import transform_train, TO_TENSOR_RESCALE
from .data.imagenet_dataloader import ImagenetDataset
from .data.modelnet_dataloader import ModelNet40


def dl(ds, bs, shuffle, drop_last):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=8, drop_last=drop_last)

def unpack(tp, conf, device):
    if conf['is_3d']:
        udf_points, distances, label = tp
        udf_points, distances, label = udf_points.to(device), distances.to(device), label.to(device)
        image = (udf_points, distances)
    else:
        image, label = tp
        image, label = image.to(device), label.to(device)
    return image, label


def fetch_dataset(conf):

    conf['is_3d'] = False
    conf['interval_log'] = 64
    conf['interval_preview'] = 512
    test_batch_size = 32

    dataset = conf['dataset']
    data_path = os.path.join(conf['base_path'], "raw")

    if dataset == 'modelnet40':
        # stage, transform, cloud_size, num_steps, num_udf
        H, W = None, None
        test_batch_size = 8
        resolution = (32 * 1024)
        num_udf = int(resolution * conf['subsample_points'])
        conf['sample_size'] = num_udf
        dataset_train = ModelNet40(stage='train', transform='train' if conf['augmentations'] else 'none',
                                cloud_size=resolution, num_steps=conf['inner_steps']+1, num_udf=num_udf)
        dataset_val_train = torch.utils.data.Subset(dataset_train, torch.randperm(len(dataset_train))[:500])
        dataset_val_val = ModelNet40(stage='test', transform='val',
                                cloud_size=resolution, num_steps=conf['inner_steps']+1, num_udf=num_udf)
        conf['is_3d'] = True
        conf['class_count'] = 40
        conf['color_channels'] = 1 # outputs single distance
        if conf['epochs'] == 0:
            conf['epochs'] = 150
        conf['omega'] = 10.0

    elif dataset == 'cifar10':
        H, W = 32, 32
        conf['class_count'] = 10
        conf['color_channels'] = 3

        if conf['augmentations']:
            t_transf = transform_train(image_size=(H,W), first_resize=False, flip=True, rescale=True)
        else:
            t_transf = TO_TENSOR_RESCALE

        dataset_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=t_transf)
        dataset_val_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=t_transf)
        dataset_val_val = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=TO_TENSOR_RESCALE)
        if conf['epochs'] == 0:
            conf['epochs'] = 10
        conf['omega'] = 10.0

    elif dataset == 'cifar10_trainval': # take out 20% of training data and validate on that
        dataset_train_all = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=TO_TENSOR_RESCALE)
        N = len(dataset_train_all)
        torch.manual_seed(8) # always use same split
        rand = torch.randperm(N)
        dataset_train = torch.utils.data.Subset(dataset_train_all, rand[:int(0.8*N)])
        dataset_val_train = dataset_train # same as train
        dataset_val_val = torch.utils.data.Subset(dataset_train_all, rand[int(0.8*N):])
        
        conf['class_count'] = 10
        conf['color_channels'] = 3
        H, W = 32, 32
        if conf['epochs'] == 0:
            conf['epochs'] = 10
        conf['omega'] = 10.0

    elif dataset == 'mnist':

        H, W = 28, 28
        if conf['augmentations']:
            t_transf = transform_train(image_size=(H, W), first_resize=False, flip=False, rescale=True)
        else:
            t_transf = TO_TENSOR_RESCALE
            
        dataset_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=t_transf)
        dataset_val_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=t_transf)
        dataset_val_val = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=TO_TENSOR_RESCALE)
        conf['class_count'] = 10
        conf['color_channels'] = 1
        if conf['epochs'] == 0:
            conf['epochs'] = 10
        conf['omega'] = 10.0

    elif dataset == 'fashionmnist':
        conf['class_count'] = 10
        conf['color_channels'] = 1
        H, W = 28, 28

        if conf['augmentations']:
            t_transf = transform_train(image_size=(H, W), first_resize=False, flip=True, rescale=True)
        else:
            t_transf = TO_TENSOR_RESCALE
            
        dataset_train = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=t_transf)
        dataset_val_train = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=t_transf)
        dataset_val_val = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=TO_TENSOR_RESCALE)

        if conf['epochs'] == 0:
            conf['epochs'] = 10
        conf['omega'] = 10.0

    elif dataset == 'imagenette':
        conf['class_count'] = 10
        conf['color_channels'] = 3
        H, W = 320, 320

        IMAGNETTE_TRAIN = 'DEFINE_PATH_HERE'
        IMAGNETTE_VAL = 'DEFINE_PATH_HERE'

        dataset_train = ImagenetDataset(transform='train' if conf['augmentations'] else 'val', root=IMAGNETTE_TRAIN)
        dataset_val_train = torch.utils.data.Subset(dataset_train, torch.randperm(len(dataset_train))[:500])
        dataset_val_val = ImagenetDataset(transform='val', root=IMAGNETTE_VAL)
        if conf['epochs'] == 0:
            conf['epochs'] = 40
        conf['omega'] = 30.0

    elif dataset == 'imagenet':
        test_batch_size = 16
        conf['class_count'] = 1000
        conf['color_channels'] = 3
        H, W = 320, 320

        IMGNET_TRAIN = 'DEFINE_PATH_HERE'
        IMGNET_VAL = 'DEFINE_PATH_HERE'
        dataset_train = ImagenetDataset(transform='train' if conf['augmentations'] else 'val', root=IMGNET_TRAIN)
        dataset_val_train = torch.utils.data.Subset(dataset_train, torch.randperm(len(dataset_train))[:2000])
        dataset_val_val = ImagenetDataset(transform='val', root=IMGNET_VAL)
        if conf['epochs'] == 0:
            conf['epochs'] = 40
        conf['omega'] = 30.0
        conf['interval_preview'] *= 16

    print('train size', len(dataset_train))
    print('val size', len(dataset_val_val))
    
    if conf['is_3d']:
        pass
    else:
        # assert this is actual image size
        CH, CW = dataset_train[0][0].shape[1:] # [C, H, W]
        assert (H, W) == (CH, CW), f'expected {H}x{W} but got {CH}x{CW}'
        conf['sample_size'] = int(H*W * conf['subsample_points'])

    print('[IMPORTANT] Sampling pixels per step', conf['sample_size'])

    loader_train = dl(ds=dataset_train, bs=conf['batch_size'], shuffle=True, drop_last=True)
    loader_val_train = dl(dataset_val_train, bs=test_batch_size, shuffle=False, drop_last=False)
    loader_val_val = dl(dataset_val_val, bs=test_batch_size, shuffle=False, drop_last=False)

    return loader_train, loader_val_train, loader_val_val