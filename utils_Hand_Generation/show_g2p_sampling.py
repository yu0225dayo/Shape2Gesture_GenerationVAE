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

def demo_multi_sampling(data, num_samples=5):
    """
    同じサンプルに対して複数回サンプリングを行う
    
    Args:
        data: データセットから取得したサンプル
        num_samples: サンプリング回数
    
    Returns:
        point_set: 点群
        pred_choice: パーツセグメンテーション結果
        hand_target: Ground Truth
        pred_ges_l_list: 左手の予測リスト (num_samples個)
        pred_ges_r_list: 右手の予測リスト (num_samples個)
    """
    print(f"########### Generating {num_samples} samples ###########")

    point_set, seglabel, hand_target, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
    point = point_set.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat = pointnet(point)
    pred_choice = pred.data.max(2)[1].cpu()

    print("推測  label 1 ,2 ,0:", np.count_nonzero(pred_choice==1), np.count_nonzero(pred_choice==2), np.count_nonzero(pred_choice==0))
    print("答え  label 1 ,2 ,0:", np.count_nonzero(seglabel==1), np.count_nonzero(seglabel==2), np.count_nonzero(seglabel==0))

    points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target = data
    
    # パーツセグメンテーション
    pl, pr, all_feat, plout, prout = get_patseg_target(pointnet, points.view(1, points.size()[0], point.size()[1]), target.view(1, target.size()[0], target.size()[1]))
    pl, pr, all_feat, plout, prout = get_patseg(pointnet, points.view(1, points.size()[0], point.size()[1]), target.view(1, target.size()[0], target.size()[1]))
    #パーツの可視化
    print("Visualizing segmented parts...")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_axis_off()
    drawpts(points.numpy(), pred_choice[0].numpy(), ax)
    
    # パーツの特徴ベクトル取得
    pf_l, mu_l, logvar_l = parts_encoder_l(pl, all_feat)
    pf_r, mu_r, logvar_r = parts_encoder_r(pr, all_feat)  
    
    # 基準の手を生成
    pred_handl = handvae_l.finetune(pf_l)
    pred_handr = handvae_r.finetune(pf_r)
    
    # 複数回サンプリング
    pred_ges_l_list = []
    pred_ges_r_list = []
    mse_l_list = []
    mse_r_list = []
    
    wrist_format = torch.tensor([0.5, 0.5, 0.5])
    pred_handl_base = pred_handl.view(1, -1, 3) - wrist_format
    pred_handr_base = pred_handr.view(1, -1, 3) - wrist_format
    
    hscale_l, hscale_r = np.split(hand_scale, 2, axis=0)
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        # 回転行列生成 (VAEなので毎回異なる結果が得られる)
        R_l, wrist_l, kld_Rl, zl = position_generater_l(plout, all_feat, N=1)
        R_r, wrist_r, kld_Rr, zr = position_generater_r(prout, all_feat, N=1)
        R_l, R_r = z_rotation_matrix(R_l).cpu().detach(), z_rotation_matrix(R_r).detach().cpu()
        
        wrist_l, wrist_r = wrist_l.view(1, -1, 3), wrist_r.view(1, -1, 3)
        pred_ges_l_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handl_base], dim=1)
        pred_ges_r_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handr_base], dim=1)
        
        pred_ges_l = pred_ges_l_format / hscale_l
        pred_ges_r = pred_ges_r_format / hscale_r

        pred_ges_l = pred_ges_l.unsqueeze(1) @ R_l.transpose(2, 3) + wrist_l.unsqueeze(2)
        pred_ges_r = pred_ges_r.unsqueeze(1) @ R_r.transpose(2, 3) + wrist_r.unsqueeze(2)
        pred_ges_l, pred_ges_r = pred_ges_l[0], pred_ges_r[0]
        
        pred_ges_l_list.append(pred_ges_l[0].detach().cpu().numpy())
        pred_ges_r_list.append(pred_ges_r[0].detach().cpu().numpy())
        
        # MSE計算
        targetl, targetr = np.split(hand_target, 2, axis=0)
        if label[:2] != "bo":  # bottle以外
            msel = F.mse_loss(pred_ges_l, targetl.view(1, 23, 3))
            mser = F.mse_loss(pred_ges_r, targetr.view(1, 23, 3))
            print(f"MSE_L: {msel.item():.6f}")
            print(f"MSE_R: {mser.item():.6f}")
            mse_l_list.append(msel.item())
            mse_r_list.append(mser.item())
    
    # 統計情報
    if len(mse_l_list) > 0:
        print("\n========== Statistics ==========")
        print(f"Left Hand  MSE - Mean: {np.mean(mse_l_list):.6f}, Std: {np.std(mse_l_list):.6f}, Min: {np.min(mse_l_list):.6f}, Max: {np.max(mse_l_list):.6f}")
        print(f"Right Hand MSE - Mean: {np.mean(mse_r_list):.6f}, Std: {np.std(mse_r_list):.6f}, Min: {np.min(mse_r_list):.6f}, Max: {np.max(mse_r_list):.6f}")
    
    return point_set, pred_choice, hand_target, pred_ges_l_list, pred_ges_r_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='save_model', help='model path')
    parser.add_argument('--idx', type=int, default=6, help='model index')
    parser.add_argument('--dataset', type=str, default='neuralnet_dataset_unity', help='dataset path')
    parser.add_argument('--select_labels', type=list, default=["ka"], help="what class use in dataset")
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples to generate')
    opt = parser.parse_args()
    print(opt)

    d = ShapeNetDataset_format_select(
        root=opt.dataset,
        data_augmentation=False,
        select_labels=opt.select_labels
    )
    data = d[opt.idx]
    point_set, seglabel, hand_target, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
    
    # partseg model
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

    print("=== [n] 次のサンプル / [r] 再サンプリング / [q] 終了 ===")
    
    # Figure作成
    plt.ion()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show(block=False)

    current_data = data

    # メインループ
    while True:
        key = input("Press [n] for next sample, [r] to resample, [q] to quit: ")

        if key.lower() == "q":
            print("Exit")
            break

        if key.lower() == "n":
            # 次のサンプル
            opt.idx += 1
            try:
                current_data = d[opt.idx]
            except:
                print("End of dataset, wrapping to start")
                opt.idx = 0
                current_data = d[opt.idx]
        
        elif key.lower() == "r":
            # 同じサンプルで再サンプリング
            pass
        
        else:
            continue

        # 推論
        point_set, pred_choice, hand_target, pred_ges_l_list, pred_ges_r_list = demo_multi_sampling(current_data, num_samples=opt.num_samples)

        # 描画更新
        for i in range(opt.num_samples):
            # point cloud
            drawparts(point_set.numpy(), ax=ax, parts="")

            # hand pose
            drawhand(pred_ges_l_list[i], ax=ax, color="orange")
            drawhand(pred_ges_r_list[i], ax=ax, color="purple")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # 終了待ち
    print("Close window to finish.")
    plt.ioff()
    plt.show()