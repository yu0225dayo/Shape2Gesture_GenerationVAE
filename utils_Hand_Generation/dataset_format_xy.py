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

def normalize_hand(hand):
    """手の大きさを正規化(0.5)"""
    middle_len = sum(np.array([np.linalg.norm(hand[0]-hand[8]),
                np.linalg.norm(hand[8]-hand[9]),
                np.linalg.norm(hand[9]-hand[10]),
                np.linalg.norm(hand[10]-hand[20])]))
    
    hscale = 1 / middle_len / 2
    hand_normalized = hand * hscale 
    return hand_normalized, hscale

def rotation_matrix_z(theta):
    """Z軸周りの2D回転行列"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

def format_hand(hand):
    """
    親指ベクトルのXY成分が(1, 0)を向くように、XY平面上で回転
    """
    vec_thumb = hand[18][:2]  # XY成分だけ
    vec_thumb = vec_thumb / np.linalg.norm(vec_thumb)

    # 現在の角度を求めて、(1, 0)との角度を算出
    angle = np.arctan2(vec_thumb[1], vec_thumb[0])  # Y, X

    # -angle回転すれば、(1, 0)になる
    R = rotation_matrix_z(-angle)

    hand_format = hand @ R.T  # 3D点群に3x3回転を適用

    return hand_format, R.T, vec_thumb


class ShapeNetDataset_format(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 data_augmentation=True, 
                 theta=None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        datafol = os.path.join(self.root, split)
        self.theta = theta
        print(datafol)
        self.datapath = []
        namefol = os.path.join(datafol, "pts")
        
        self.cate = []
        self.cate_num = []

        # すべてのクラスが同じサンプル数になるように
        if self.split == "train":
            for csv in os.listdir(namefol):
                if self.cate == [] or self.cate[-1][:2] != csv[:2]:
                    self.cate.append(csv[:2])
                    self.cate_num.append(1)
                else:
                    self.cate_num[-1] += 1
            # リピート回数
            self.repeat = [(max(self.cate_num) // cate_num) for cate_num in self.cate_num]
            dict_fol = dict(zip(self.cate, self.repeat))
            # データ数を合わせる
            for csv in os.listdir(namefol):
                k = dict_fol[csv[:2]]
                for j in range(k):
                    pts = os.path.join(datafol, "pts", csv)
                    csvname, ext = os.path.splitext(csv)
                    label = os.path.join(datafol, "label", csvname + "_label" + ext)
                    hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
                    self.datapath.append([csv[:2], pts, label, hand, csvname])
                    
        else:
            for csv in os.listdir(namefol):
                pts = os.path.join(datafol, "pts", csv)
                csvname, ext = os.path.splitext(csv)
                label = os.path.join(datafol, "label", csvname + "_label" + ext)
                hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
                self.datapath.append([csv[:2], pts, label, hand, csvname])

        print("samples of data:", len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        filename = fn[4]
        point_set = np.array(pd.read_csv(fn[1], header=None)).astype(np.float32)
        hand = np.array(pd.read_csv(fn[3], header=None)).astype(np.float32)
        
        seg = pd.read_csv(fn[2], header=None).to_numpy().astype(np.int64)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        hand_set_rote = hand / dist  # object scale
        
        if self.data_augmentation and filename[:2] != "bo":  # ボトルクラスなら回転させない
            if self.theta == None:
                theta = np.random.uniform(0, np.pi * 2)
            else:
                theta = np.random.uniform(-self.theta, self.theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                       [np.sin(theta), np.cos(theta)]])
            random_jitter = np.random.normal(0, 0.02, size=point_set.shape)
            point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(rotation_matrix)  # random rotation
            point_set += random_jitter  # random jitter
            hand_set_rote[:, [0, 1]] = hand_set_rote[:, [0, 1]].dot(rotation_matrix)

        hand_l, hand_r = np.split(hand_set_rote, 2, axis=0)
        wrist_l, wrist_r = hand_l[0], hand_r[0]
        wrist = np.vstack((wrist_l, wrist_r))

        # 中指の長さが0.5になるように調整する。後に手首座標系に変換するときに0~1になる。
        hand_l, hscale_l = normalize_hand(hand_l)
        hand_r, hscale_r = normalize_hand(hand_r)
        hand_set = np.vstack((hand_l, hand_r))

        # format hand 
        # 手首座標系に変換
        hand_l, hand_r = np.split(hand_set, 2, axis=0)
        hand_l = hand_l - hand_l[0]
        hand_r = hand_r - hand_r[0]
        hand_l_format, sita_l, _ = format_hand(hand_l) 
        hand_r_format, sita_r, _ = format_hand(hand_r) 

        hand_format = np.vstack((hand_l_format, hand_r_format))
        sita_ans = np.vstack((sita_l.T, sita_r.T))
        hand_scale = np.vstack((hscale_l, hscale_r))

        seg = seg[choice]
        batch_weight = np.array([np.count_nonzero(seg == 0), 
                                np.count_nonzero(seg == 1),
                                np.count_nonzero(seg == 2)]) / len(seg)
        
        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg)
        hand_set_rote = torch.from_numpy(hand_set_rote.astype(np.float32))
        batch_weight = torch.from_numpy(batch_weight.astype(np.float32))
        hand_set = torch.from_numpy(hand_set.astype(np.float32))
        hand_scale = torch.from_numpy(hand_scale.astype(np.float32))
        hand_format = torch.from_numpy(hand_format.astype(np.float32))
        sita_ans = torch.from_numpy(sita_ans.astype(np.float32))
        wrist = torch.from_numpy(wrist.astype(np.float32))

        return point_set, seg, hand_set_rote, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist
          
    def __len__(self):
        return len(self.datapath)


class ShapeNetDataset_format_select(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 data_augmentation=True,
                 theta=None,
                 select_labels=None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.theta = theta
        self.select_labels = select_labels

        datafol = os.path.join(self.root, split)
        namefol = os.path.join(datafol, "pts")
        print(datafol)

        self.datapath = []
        self.cate = []
        self.cate_num = []

        if self.split == "train":
            for csv in os.listdir(namefol):
                label_id = csv[:2]
                if self.select_labels is not None:
                    if label_id not in self.select_labels:
                        continue
                if self.cate == [] or self.cate[-1] != label_id:
                    self.cate.append(label_id)
                    self.cate_num.append(1)
                else:
                    self.cate_num[-1] += 1

            max_num = max(self.cate_num)
            self.repeat = [(max_num // n) for n in self.cate_num]
            dict_fol = dict(zip(self.cate, self.repeat))

            for csv in os.listdir(namefol):
                label_id = csv[:2]
                if self.select_labels is not None:
                    if label_id not in self.select_labels:
                        continue
                k = dict_fol[label_id]
                for _ in range(k):
                    pts = os.path.join(datafol, "pts", csv)
                    csvname, ext = os.path.splitext(csv)
                    label = os.path.join(datafol, "label", csvname + "_label" + ext)
                    hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
                    self.datapath.append([label_id, pts, label, hand, csvname])
        else:
            for csv in os.listdir(namefol):
                label_id = csv[:2]
                if self.select_labels is not None:
                    if label_id not in self.select_labels:
                        continue
                pts = os.path.join(datafol, "pts", csv)
                csvname, ext = os.path.splitext(csv)
                label = os.path.join(datafol, "label", csvname + "_label" + ext)
                hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
                self.datapath.append([label_id, pts, label, hand, csvname])

        print("use classes:", self.select_labels)
        print("samples of data:", len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        filename = fn[4]
        point_set = np.array(pd.read_csv(fn[1], header=None)).astype(np.float32)
        hand = np.array(pd.read_csv(fn[3], header=None)).astype(np.float32)
        seg = pd.read_csv(fn[2], header=None).to_numpy().astype(np.int64)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        point_set = point_set - np.mean(point_set, axis=0, keepdims=True)
        dist = np.max(np.linalg.norm(point_set, axis=1))
        point_set = point_set / dist
        hand_set_rote = hand / dist

        if self.data_augmentation and filename[:2] != "bo":
            if self.theta is None:
                theta = np.random.uniform(0, np.pi * 2)
            else:
                theta = np.random.uniform(-self.theta, self.theta)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            jitter = np.random.normal(0, 0.02, size=point_set.shape)
            point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(R)
            point_set += jitter
            hand_set_rote[:, [0, 1]] = hand_set_rote[:, [0, 1]].dot(R)

        hand_l, hand_r = np.split(hand_set_rote, 2, axis=0)
        wrist = np.vstack((hand_l[0], hand_r[0]))

        hand_l, hscale_l = normalize_hand(hand_l)
        hand_r, hscale_r = normalize_hand(hand_r)
        hand_set = np.vstack((hand_l, hand_r))
        hand_scale = np.vstack((hscale_l, hscale_r))

        hand_l, hand_r = np.split(hand_set, 2, axis=0)
        hand_l -= hand_l[0]
        hand_r -= hand_r[0]

        hand_l_format, sita_l, _ = format_hand(hand_l)
        hand_r_format, sita_r, _ = format_hand(hand_r)
        hand_format = np.vstack((hand_l_format, hand_r_format))
        sita_ans = np.vstack((sita_l.T, sita_r.T))

        seg = seg[choice]
        batch_weight = np.array([
            np.count_nonzero(seg == 0),
            np.count_nonzero(seg == 1),
            np.count_nonzero(seg == 2)
        ]) / len(seg)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg)
        hand_set_rote = torch.from_numpy(hand_set_rote.astype(np.float32))
        batch_weight = torch.from_numpy(batch_weight.astype(np.float32))
        hand_set = torch.from_numpy(hand_set.astype(np.float32))
        hand_scale = torch.from_numpy(hand_scale.astype(np.float32))
        hand_format = torch.from_numpy(hand_format.astype(np.float32))
        sita_ans = torch.from_numpy(sita_ans.astype(np.float32))
        wrist = torch.from_numpy(wrist.astype(np.float32))

        return point_set, seg, hand_set_rote, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist
    def __len__(self):
        return len(self.datapath)


class ShapeNetDataset2(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 data_augmentation=True,
                 theta=None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        datafol = os.path.join(self.root, split)
        self.theta = theta
        print(datafol)
        self.datapath = []
        namefol = os.path.join(datafol, "pts")
        
        self.cate = []
        self.cate_num = []

        if self.split == "train":
            for csv in os.listdir(namefol):
                if self.cate == [] or self.cate[-1][:2] != csv[:2]:
                    self.cate.append(csv[:2])
                    self.cate_num.append(1)
                else:
                    self.cate_num[-1] += 1
            self.repeat = [(max(self.cate_num) // cate_num) for cate_num in self.cate_num]
            dict_fol = dict(zip(self.cate, self.repeat))
            for csv in os.listdir(namefol):
                k = dict_fol[csv[:2]]
                for j in range(k):
                    pts = os.path.join(datafol, "pts", csv)
                    csvname, ext = os.path.splitext(csv)
                    label = os.path.join(datafol, "label", csvname + "_label" + ext)
                    hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
                    self.datapath.append([csv[:2], pts, label, hand, csvname])
        else:
            for csv in os.listdir(namefol):
                pts = os.path.join(datafol, "pts", csv)
                csvname, ext = os.path.splitext(csv)
                label = os.path.join(datafol, "label", csvname + "_label" + ext)
                hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
                self.datapath.append([csv[:2], pts, label, hand, csvname])

        print("samples of data:", len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        filename = fn[4]
        point_set = np.array(pd.read_csv(fn[1], header=None)).astype(np.float32)
        hand = np.array(pd.read_csv(fn[3], header=None)).astype(np.float32)
        seg = pd.read_csv(fn[2], header=None).to_numpy().astype(np.int64)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist
        hand_set_rote = hand / dist

        if self.data_augmentation and filename[:2] != "bo":
            if self.theta is None:
                theta = np.random.uniform(0, np.pi * 2)
            else:
                theta = np.random.uniform(-self.theta, self.theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
            random_jitter = np.random.normal(0, 0.02, size=point_set.shape)
            point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(rotation_matrix)
            point_set += random_jitter
            hand_set_rote[:, [0, 1]] = hand_set_rote[:, [0, 1]].dot(rotation_matrix)

        hand_l, hand_r = np.split(hand_set_rote, 2, axis=0)
        wrist_l, wrist_r = hand_l[0], hand_r[0]
        wrist = np.vstack((wrist_l, wrist_r))

        hand_l, hscale_l = normalize_hand(hand_l)
        hand_r, hscale_r = normalize_hand(hand_r)
        hand_set = np.vstack((hand_l, hand_r))

        hand_l, hand_r = np.split(hand_set, 2, axis=0)
        hand_l = hand_l - hand_l[0]
        hand_r = hand_r - hand_r[0]

        hand_l_format, sita_l, _ = format_hand(hand_l)
        hand_r_format, sita_r, _ = format_hand(hand_r)
        hand_format = np.vstack((hand_l_format, hand_r_format))
        sita_ans = np.vstack((sita_l.T, sita_r.T))
        hand_scale = np.vstack((hscale_l, hscale_r))

        seg = seg[choice]
        batch_weight = np.array([np.count_nonzero(seg == 0),
                                np.count_nonzero(seg == 1),
                                np.count_nonzero(seg == 2)]) / len(seg)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg)
        hand_set_rote = torch.from_numpy(hand_set_rote.astype(np.float32))
        batch_weight = torch.from_numpy(batch_weight.astype(np.float32))
        hand_set = torch.from_numpy(hand_set.astype(np.float32))
        hand_scale = torch.from_numpy(hand_scale.astype(np.float32))
        hand_format = torch.from_numpy(hand_format.astype(np.float32))
        sita_ans = torch.from_numpy(sita_ans.astype(np.float32))
        wrist = torch.from_numpy(wrist.astype(np.float32))

        return point_set, seg, hand_set_rote, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist

    def __len__(self):
        return len(self.datapath)


class HandDataset_format(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 data_augmentation=True,
                 theta=None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.theta = theta
        datafol = os.path.join(self.root, split)
        print(datafol)
        
        self.datapath = []
        for csv in os.listdir(datafol):
            csvname, ext = os.path.splitext(csv)
            hand = os.path.join(datafol, csvname + ext)
            self.datapath.append([csv[:2], hand, csvname])

    def __getitem__(self, index):
        fn = self.datapath[index]
        filename = fn[2]
        hand_set = np.array(pd.read_csv(fn[1], header=None)).astype(np.float32)
        hand_l, hand_r = np.split(hand_set, 2, axis=0)

        hand_l, hscale_l = normalize_hand(hand_l)
        hand_r, hscale_r = normalize_hand(hand_r)
        hand_set = np.vstack((hand_l, hand_r))
        hand_set_rote = hand_set.copy()

        if self.data_augmentation and filename[:2] != "bo":
            if self.theta is None:
                theta = np.random.uniform(0, np.pi * 2)
            else:
                theta = np.random.uniform(-self.theta, self.theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
            hand_set_rote[:, [0, 1]] = hand_set_rote[:, [0, 1]].dot(rotation_matrix)

        hand_l, hand_r = np.split(hand_set_rote, 2, axis=0)
        wrist_l, wrist_r = hand_l[0], hand_r[0]
        wrist = np.vstack((wrist_l, wrist_r))

        hand_l = hand_l - hand_l[0]
        hand_r = hand_r - hand_r[0]

        hand_l_format, sita_l, _ = format_hand(hand_l)
        hand_r_format, sita_r, _ = format_hand(hand_r)
        hand_format = np.vstack((hand_l_format, hand_r_format))
        sita_ans = np.vstack((sita_l.T, sita_r.T))
        hand_scale = np.vstack((hscale_l, hscale_r))

        hand_set_rote = torch.from_numpy(hand_set_rote.astype(np.float32))
        hand_set = torch.from_numpy(hand_set.astype(np.float32))
        hand_format = torch.from_numpy(hand_format.astype(np.float32))
        sita_ans = torch.from_numpy(sita_ans.astype(np.float32))
        wrist = torch.from_numpy(wrist.astype(np.float32))
        hand_scale = torch.from_numpy(hand_scale.astype(np.float32))

        return hand_set_rote, hand_set, hand_format, sita_ans, wrist, hand_scale

    def __len__(self):
        return len(self.datapath)