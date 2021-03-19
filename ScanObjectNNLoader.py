import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

from pointnet2_ops import pointnet2_utils
from util import index_points

from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class ScanObjectNNCls(data.Dataset):

    def __init__(
            self, transforms=None, train=True, self_supervision=False, num_points=2048,
    ):
        super().__init__()

        self.transforms = transforms

        self.self_supervision = self_supervision

        self.train = train

        self.num_points = num_points

        root = '/home/zhangniansong/data/ScanObjNN/main_split_nobg/'
        if self.self_supervision:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            points_train = np.array(h5['data']).astype(np.float32)
            h5.close()
            self.points = points_train
            self.labels = None
        elif train:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            h5 = h5py.File(root + 'test_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()

        self.points = torch.tensor(self.points).cuda() # maybe modify to `to(device)`
        fps_idx = pointnet2_utils.furthest_point_sample(self.points, self.num_points)
        self.points = index_points(self.points, fps_idx.long())

        print('Successfully load ScanObjectNN with', len(self.labels), 'instances')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].clone()
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        if self.self_supervision:
            return current_points
        else:
            label = self.labels[idx]
            return current_points, label

    def __len__(self):
        return self.points.shape[0]

if __name__ == '__main__':
    trainset = ScanObjectNNCls(num_points=1024)
    train_loader = DataLoader(trainset, batch_size=2,shuffle=True, drop_last=True)
    for data, label in train_loader:
        import ipdb; ipdb.set_trace()
        print(label.shape)


