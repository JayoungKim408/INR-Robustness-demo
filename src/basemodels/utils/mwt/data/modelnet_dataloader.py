# create dataloader

import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
# import trimesh
import math
from scipy.spatial import cKDTree

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

PATH = 'DEFINE_PATH_HERE'


def load():
    # first find classes by listing all directories in the path
    classes = []
    for dir in os.listdir(PATH):
        if os.path.isdir(PATH + dir):
            classes.append(dir)
    assert len(classes) == 40, 'Found ' + str(len(classes)) + ' classes, expected 40.'
    classes.sort() # sort them alphabetically
    print('found ModelNet40 classes:', classes)
    # each class contains train and test dir
    return classes


def read_off(filename):
    with open(filename, 'r') as file:
        header = file.readline().strip()
        if not header.startswith('OFF'):
            raise Exception('Not a valid OFF header', header, 'but expected OFF in file', filename)
        leftover = header[3:]
        if leftover == '':
            leftover = file.readline().strip()
        n_verts, n_faces, _ = tuple([int(s) for s in leftover.split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        # assert file is empty now
        assert file.readline() == '', 'Expected empty file but found more content'
        # assert nonzero number of verts and faces
        assert n_verts > 0 and n_faces > 0, 'Expected more than 0 vertices and faces'
        return verts, faces


class ModelNet40(Dataset):

    def __init__(self, stage, transform, cloud_size, num_steps, num_udf):
        super().__init__()
        self.transform = transform
        self.cloud_size = cloud_size
        self.num_steps = num_steps
        self.num_udf = num_udf

        classes = load()
        self.labels = classes
        files = []
        for clazz in classes:
            path = PATH + clazz + '/' + stage + '/'
            # list all files and loop
            for file in os.listdir(path):
                # clazz must be in file name
                assert clazz in file, 'Class name not in file name'
                pair = (path + file, clazz)
                files.append(pair)
        self.files = files
        print('Found', len(self.files), 'files for ModelNet40 dataset loader and a total of', len(self.labels), 'classes.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path_name, label = self.files[idx]
        label = self.labels.index(label)
        verts, faces = read_off(path_name)
        # verts = torch.tensor(verts) # [382, 3]
        # faces = torch.tensor(faces) # [758, 3]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        points, _ = trimesh.sample.sample_surface(mesh, self.cloud_size) # [P, XYZ=3]
        points = torch.tensor(points)

        # normalize max [-1, +1] and zero mean
        points = points - torch.mean(points, dim=0, keepdim=True) # subtract the X, Y, Z means
        points = points / torch.amax(torch.abs(points)) # divide by the max value
        points = points.float()

        if self.transform == 'train':
            # random scaling
            points = points * (0.9 + 0.2 * torch.rand(1)) # scale between 0.8 and 1.2

            # pointwise jitter
            points = points + 0.01 * torch.randn_like(points) # add noise

        elif self.transform == 'val' or self.transform == 'none':
            # no augmentation
            pass
        else:
            raise ValueError('transform should be either train or val')

        # convert to UDF values
        udf_points = torch.rand(size=(self.num_steps * self.num_udf, 3)).float() * 2 - 1 # [ST, P_UDF, XYZ=3]
        points = points.numpy()
        udf_points = udf_points.numpy()
        kd_tree = cKDTree(points)
        distances, _ = kd_tree.query(udf_points, k=1) # [ST, P_UDF]
        distances = torch.tensor(distances).float() # [ST*P_UDF]
        distances = distances.view(self.num_steps, self.num_udf, 1) # [ST, P_UDF]
        udf_points = torch.tensor(udf_points).float()
        udf_points = udf_points.view(self.num_steps, self.num_udf, 3)
        return udf_points, distances, label # [ST, P_UDF, XYZ=3], [ST, P_UDF, 1]
