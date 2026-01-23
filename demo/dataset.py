from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#実世界の点群データセット
class RealWorld_Dataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 theta=None):
        self.npoints = npoints
        self.root = root
        self.split = split
        datafol = os.path.join(self.root, split)
        self.theta = theta
        print(datafol)
        self.datapath = []
        namefol = os.path.join(datafol, "pts")
        for csv in os.listdir(namefol):
            pts = os.path.join(datafol, "pts", csv)
            csvname, ext = os.path.splitext(csv)
            self.datapath.append([csv[:2], pts, csvname])
        print("samples of data:", len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        filename = fn[2]
        point_set = np.array(pd.read_csv(fn[1], header=None)).astype(np.float32)
        print(len(point_set))
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        point_set = torch.from_numpy(point_set.astype(np.float32))
        return point_set, filename
          
    def __len__(self):
        return len(self.datapath)