from .mnist_inr_dataset import LabeledINRDataset, MNISTINRImageDataset, MNISTINRDataset
from .fmnist_inr_dataset import LabeledFashionMnistINRDataset, FashionMNISTINRDataset
from .cifar10_dataset import NFNZooDataset, CNNDataset
from .cifar_inr_dataset import CifarINRDataset
from .shapenet10_inr_dataset import ShapeNet10INRDataset
from .perturbed_dataset import INRPerturbedDataset


def dataset(dataset_config, **kwargs):
    _map = {
        'inr_perturbed': INRPerturbedDataset,
        'labeled_mnist_inr': LabeledINRDataset,
        'labeled_fashion_mnist_inr': LabeledFashionMnistINRDataset,
        'cifar_inr': CifarINRDataset,
        'mnist_inr': MNISTINRDataset,
        'fashion_inr': FashionMNISTINRDataset,
        'shapenet_inr': ShapeNet10INRDataset,
        'cifar10': NFNZooDataset,
        'svhn': NFNZooDataset,
        'mnist_inr_edit': MNISTINRImageDataset
    }
    # return _map[dataset_config["dataset"]](**dataset_config, **kwargs)
    return _map[dataset_config["dataset"]](**dataset_config, **kwargs)