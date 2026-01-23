#ges2ges　pretrained model AE
from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset_format_xy import ShapeNetDataset_format, HandDataset_format
#from model2 import PointNetDenseCls, feature_transform_regularizer , ContrastiveNet, PartsToPtsNet2, DecoderFromPf
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import shutil
import matplotlib.pyplot as plt
from visualization import *
from model import HandVAE

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='save_model/pretrained_HandVAE_formatxy', help='model path')
parser.add_argument('--idx', type=int, default=150, help='model index')
parser.add_argument('--dataset', type=str, default='neuralnet_dataset_unity', help='dataset path')

opt = parser.parse_args()
print(opt)

handvae_l, handvae_r = HandVAE(), HandVAE()
state_dict_vae_l = torch.load(f"{opt.model}/vae_l_best.pth", weights_only = True)
state_dict_vae_r = torch.load(f"{opt.model}/vae_r_best.pth", weights_only = True)

handvae_l.load_state_dict(state_dict_vae_l)
handvae_l.eval()
handvae_r.load_state_dict(state_dict_vae_r)
handvae_r.eval()

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax.set_title('target_hand_l', fontsize=10) 

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax1.set_title('target_hand_r', fontsize=10) 

ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(-1,1)

fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax2.set_title('pred_hand_l', fontsize=10) 

ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(-1,1)

fig3, ax3 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax3.set_title('pred_hand_r', fontsize=10) 

ax3.set_xlim(-1,1)
ax3.set_ylim(-1,1)
ax3.set_zlim(-1,1)

fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax4.set_title('target vs pred', fontsize=10)

ax4.set_xlim(-1,1)
ax4.set_ylim(-1,1)
ax4.set_zlim(-1,1)

fig5, ax5 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax5.set_title('target vs pred', fontsize=10)

ax5.set_xlim(-1,1)
ax5.set_ylim(-1,1)
ax5.set_zlim(-1,1)

ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.set_zlabel("z")

def test_hand_dataset():

    d = HandDataset_format(
    root = opt.dataset,
    split = 'train',
    data_augmentation = True)

    input_hands, nomalhand, hand_format = d[opt.idx]

    target_hand_l, target_hand_r = np.split(hand_format + torch.tensor([0.5, 0.5, 0.5]), 2, axis = 0) 
    target_hand_l, target_hand_r = target_hand_l.reshape(1, 69), target_hand_r.reshape(1, 69) 
    target_hand_l, target_hand_r = target_hand_l[:, 3:], target_hand_r[:, 3:]

    hand=np.split(hand_format,2,axis=0)
    hand_l = hand[0]-hand[0][0] + torch.tensor([0.5, 0.5, 0.5])
    hand_r = hand[1]-hand[1][0] + torch.tensor([0.5, 0.5, 0.5])

    hand_l=hand_l.reshape(1,69)
    hand_r=hand_r.reshape(1,69)

    hand_l_in = hand_l[:,3:]
    hand_r_in = hand_r[:,3:]

    pred_l, _ , _ = handvae_l(hand_l_in)
    pred_r, _ , _ = handvae_r(hand_r_in)

    lossl, msel, _ = handvae_l.loss(hand_l_in)
    lossr, mser, _ = handvae_r.loss(hand_r_in)
    #print("label:", label)
    print("------------")
    print(F.mse_loss(pred_l,hand_l_in,reduction="sum"))
    print(F.mse_loss(pred_r,hand_r_in,reduction="sum"))

    pred_l, pred_r = pred_l.detach().numpy() , pred_r.detach().numpy() 

    ges_l = pred_l.reshape(22,3) - [0.5,0.5,0.5] 
    ges_r = pred_r.reshape(22,3) - [0.5,0.5,0.5]

    ges_l =np.vstack(([0,0,0],ges_l)).reshape(23,3)
    ges_r =np.vstack(([0,0,0],ges_r)).reshape(23,3)

    print("msel :",msel)
    print("mser :",mser)

    #print("ges_l:", ges_l)
    drawhand(hand=(hand[0]-hand[0][0]),color="red",ax=ax,handinf=handinf)
    drawhand(hand=(hand[1]-hand[1][0]),color="blue",ax=ax1,handinf=handinf)

    drawhand(hand=ges_l,color="orange",ax=ax2,handinf=handinf)
    drawhand(hand=ges_r,color="purple",ax=ax3,handinf=handinf)

    drawhand(hand=(hand[0]-hand[0][0]),color="red",ax=ax4,handinf=handinf)
    drawhand(hand=(hand[1]-hand[1][0]),color="blue",ax=ax5,handinf=handinf)

    drawhand(hand=ges_l,color="orange",ax=ax4,handinf=handinf)
    drawhand(hand=ges_r,color="purple",ax=ax5,handinf=handinf)

    ax4.quiver(0, 0, 0, *hand[0][18] - hand[0][0], color='blue', linestyle='dashed', label='Target (1,0,0)')
    ax4.quiver(0, 0, 0, *ges_l[18], color='purple', linestyle='dashed', label='Target (1,0,0)')
    ax4.quiver(0, 0, 0, *np.array([1,0,0]), color='green', linestyle='dashed', label='Target (1,0,0)')

    ax5.quiver(0, 0, 0, *hand[1][18] - hand[1][0], color='blue', linestyle='dashed', label='Target (1,0,0)')
    ax5.quiver(0, 0, 0, *ges_l[18], color='purple', linestyle='dashed', label='Target (1,0,0)')
    ax5.quiver(0, 0, 0, *np.array([1,0,0]), color='green', linestyle='dashed', label='Target (1,0,0)') 

    plt.show()


if __name__ =="__main__":
    test_hand_dataset()
