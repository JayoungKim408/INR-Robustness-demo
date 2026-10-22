# create dataloader

import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
# import torchvision.transforms as v2
from torchvision.transforms.functional import InterpolationMode

transform_train = transforms.Compose([
    transforms.Resize((320, 320), interpolation=InterpolationMode.BILINEAR),
    v2.Pad(40, padding_mode='reflect'),
    v2.RandomRotation(degrees=30, interpolation=InterpolationMode.BILINEAR),
    v2.CenterCrop(size=(320, 320)),
    v2.RandomResizedCrop(size=(320, 320), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),    
    v2.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((320, 320), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


class ImagenetDataset(Dataset):

    def __init__(self, transform, root):
        super().__init__()
        self.transform = transform

        labels = []
        files = []
        for j, dir in enumerate(os.listdir(root)):
            if dir.startswith('n'):
                if dir not in labels:
                    labels.append(dir)

                for file in os.listdir(root + dir):
                    path = root + dir + '/' + file
                    pair = (path, dir)
                    files.append(pair)
        self.files = files
        self.labels = labels
        self.labels.sort() # sort them alphabetically
        print('labels sorted are', self.labels)
        print('Found', len(self.files), 'files for imagenette dataset loader and a total of', len(self.labels), 'classes.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name, label = self.files[idx]
        assert label in self.labels, 'label not found in labels'
        label = self.labels.index(label)
        assert label >= 0 and label < 1000, 'label should be in range 0-999'
        image = Image.open(img_name).convert("RGB")
        if self.transform == 'train':
            image = transform_train(image)
        elif self.transform == 'val' or self.transform == 'none':
            image = transform_val(image)
        else:
            raise ValueError('transform should be either train or val')

        image = image * 2 - 1
        return image, label
