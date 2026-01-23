from __future__ import print_function
import argparse
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter

from model import HandVAE, PartsEncoder_w_TNet, Position_Generater_VAE
from dataset_format_xy import ShapeNetDataset_format
from model_pointnet import PointNetDenseCls, feature_transform_regularizer
from train_positionVAE_sekitori import *
from visualization import *
from functions_pointnet_demo import *
matplotlib.use("TkAgg")

def demo(data):
    print("###########")

    point_set, seglabel, hand_target, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
    point = point_set.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat= pointnet(point)
    pred_choice = pred.data.max(2)[1].cpu()

    #pred_choice = pred_choice.cpu().data.numpy()
    #label 1=右手　2 = 左手
    print("推測  label 1 ,2 ,0:",np.count_nonzero(pred_choice==1),np.count_nonzero(pred_choice==2),np.count_nonzero(pred_choice==0))
    print("答え  label 1 ,2 ,0:",np.count_nonzero(seglabel==1),np.count_nonzero(seglabel==2),np.count_nonzero(seglabel==0))

    points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target = data
    
    #pl, pr, all_feat, plout, prout = get_patseg(pointnet, points.view(1, points.size()[0], point.size()[1]), target.view(1, target.size()[0], target.size()[1]))
    pl, pr, all_feat, plout, prout = get_patseg_target(pointnet, points.view(1, points.size()[0], point.size()[1]), target.view(1, target.size()[0], target.size()[1]))
    #パーツの特徴ベクトル取得
    pf_l, mu_l, logvar_l = parts_encoder_l(pl, all_feat)
    pf_r, mu_r, logvar_r = parts_encoder_r(pr, all_feat)  
    #基準の手を生成
    pred_handl = handvae_l.finetune(pf_l)
    pred_handr = handvae_r.finetune(pf_r)
    #回転行列生成
    R_l, wrist_l, kld_Rl, zl = position_generater_l(plout, all_feat, N=1)
    R_r, wrist_r, kld_Rr, zr = position_generater_r(prout, all_feat, N=1)
    R_l, R_r = z_rotation_matrix(R_l).cpu().detach().cpu(), z_rotation_matrix(R_r).detach().cpu()
    wrist_format = torch.tensor([0.5, 0.5, 0.5])
    pred_handl, pred_handr = pred_handl.view(1, -1, 3) - wrist_format , pred_handr.view(1, -1, 3) - wrist_format
    #print("wrist:", wrist_l, wrist_r)
    wrist_l, wrist_r = wrist_l.view(1, -1, 3), wrist_r.view(1, -1, 3)
    pred_ges_l_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handl], dim=1)
    pred_ges_r_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handr], dim=1)
    hscale_l, hscale_r = np.split(hand_scale, 2, axis=0)
    hscale_l, hscale_r = hscale_l, hscale_r
    pred_ges_l = pred_ges_l_format / hscale_l
    pred_ges_r = pred_ges_r_format / hscale_r

    pred_ges_l = pred_ges_l.unsqueeze(1) @ R_l.transpose(2,3) + wrist_l.unsqueeze(2) # B * s * 23 * 3
    pred_ges_r = pred_ges_r.unsqueeze(1) @ R_r.transpose(2,3) + wrist_r.unsqueeze(2) # B * s * 23 * 3
    pred_ges_l, pred_ges_r = pred_ges_l[0], pred_ges_r[0]
    #mu, sigma 
    mul, sigmal = position_generater_l.get_mu_sigma(plout, all_feat)
    mur, sigmar = position_generater_r.get_mu_sigma(prout, all_feat)
    print("-----------")
    print("mul, sigmal", mul.mean(), sigmal.mean())
    print("mur, sigmar", mur.mean(), sigmar.mean())

    targetl, targetr = np.split(hand_target, 2,  axis=0)
    if label[:2] != "bo": #bottle以外
        msel = F.mse_loss(pred_ges_l, targetl.view(1,23,3))
        mser = F.mse_loss(pred_ges_r, targetr.view(1,23,3))

        print("MSE_L:", msel)
        print("MSE_R:", mser)
    
    if label[:2] == "bo": #bottle
        pred_l = pred_ges_l # 1 * N * 23 * 3
        pred_r = pred_ges_r # 1 * N * 23 * 3
        
        t_l, t_r =  targetl.unsqueeze(0), targetr.unsqueeze(0)
        target_list_l = augmentation_target(t_l.view(1,23,3), N = 12) # 1 * 23 * 3
        target_list_r = augmentation_target(t_r.view(1,23,3), N = 12)

        msel = F.mse_loss(pred_l.repeat(1,12,1,1), target_list_l, reduction="none")
        mser = F.mse_loss(pred_r.repeat(1,12,1,1), target_list_r, reduction="none")
        
        print("MSE_L:", msel.mean(dim=(2,3)), msel.sum(dim=(2,3))/69)
        print("MSE_R:", mser.mean(dim=(2,3)), mser.sum(dim=(2,3))/69)

    return point_set, pred_choice, hand_target, pred_ges_l[0].detach().cpu().numpy(), pred_ges_r[0].detach().cpu().numpy(), _, _, _, pred_ges_l_format, pred_ges_r_format

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='save_model', help='model path')
    parser.add_argument('--idx', type=int, default=0, help='model index')
    parser.add_argument('--dataset', type=str, default='dataset', help='dataset path') #dataset, dataset2
    parser.add_argument('--select_labels', type=list, default=None, help="what class use in dataset") #["ba", "bo", "ju", "ka", "mu", "pa",  "pc", "po", "va"]
    opt = parser.parse_args()
    print(opt)

    d = ShapeNetDataset_format_select(
        root=opt.dataset,
        data_augmentation=False,
        select_labels=opt.select_labels
        )
    data = d[opt.idx]
    point_set, seglabel, hand_target, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
    
    #partseg model
    pointnet = PointNetDenseCls(k=3, feature_transform=None)
    state_dict_pointnet = torch.load("save_model/pointnet/pointnet_acc_partseg_best.pth", weights_only=True)
    pointnet.load_state_dict(state_dict_pointnet)
    pointnet.eval()

    # parts encoder
    parts_encoder_l, parts_encoder_r = PartsEncoder_w_TNet(), PartsEncoder_w_TNet()
    state_parts_e_l = torch.load("save_model/pretrained_PartsEncoder/parts_encoder_l_best.pth", weights_only=True)
    state_parts_e_r = torch.load("save_model/pretrained_PartsEncoder/parts_encoder_r_best.pth", weights_only=True)
    parts_encoder_l.load_state_dict(state_parts_e_l)
    parts_encoder_l.eval()
    parts_encoder_r.load_state_dict(state_parts_e_r)
    parts_encoder_r.eval()

    #hand VAE
    handvae_r = HandVAE()
    handvae_l = HandVAE()
    state_dict_vae_l = torch.load("save_model/pretrained_HnadVAE_formatxy/vae_l_best.pth", weights_only=True)
    state_dict_vae_r = torch.load("save_model/pretrained_HnadVAE_formatxy/vae_r_best.pth", weights_only=True)
    handvae_l.load_state_dict(state_dict_vae_l)
    handvae_l.eval()
    handvae_r.load_state_dict(state_dict_vae_r)
    handvae_r.eval()

    #train model 
    "rotation matrix NN"
    position_generater_l, position_generater_r = Position_Generater_VAE(), Position_Generater_VAE()
    state_sita_l = torch.load("save_model/worst30_sampler/position_generater_l/Epoch/69_epoch.pth", weights_only=True) 
    state_sita_r = torch.load("save_model/worst30_sampler/position_generater_r/Epoch/69_epoch.pth", weights_only=True)
    position_generater_l.load_state_dict(state_sita_l)
    position_generater_r.load_state_dict(state_sita_r)
    position_generater_l.eval(), position_generater_r.eval()

    

    "hand_target : GT"

    print("=== [n] 次 / [q] 終了 ===")
    # ================================
    # Figure を1回だけ作成
    # ================================
    plt.ion()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("PointCloud + Generated Grasp Pose")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show(block=False)


    # ================================
    # メインループ
    # ================================
    while True:
        key = input("Press [n] for next, [q] to quit: ")

        if key.lower() == "q":
            print("Exit")
            break

        if key.lower() != "n":
            continue

        # ----------------------------
        # データの読み込み
        # ----------------------------
        try:
            data = d[opt.idx]
        except StopIteration:
            data = d[opt.idx]

        # ----------------------------
        # 推論
        # ----------------------------
        (
            point_set,
            pred_choice,
            hand_target,
            pred_ges_l,
            pred_ges_r,
            _, _, _,
            _, _
        ) = demo(data)

        # ----------------------------
        # 描画更新（windowは消えない）
        # ----------------------------
        ax.cla()

        ax.set_title("PointCloud + Generated Grasp Pose")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # point cloud
        drawparts(point_set.numpy(), ax=ax, parts="")

        # hand pose
        drawhand(pred_ges_l, ax=ax, color="orange")
        drawhand(pred_ges_r, ax=ax, color="purple")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()


    # ================================
    # 終了待ち（ウィンドウ残す）
    # ================================
    print("Close window to finish.")
    plt.ioff()
    plt.show()
