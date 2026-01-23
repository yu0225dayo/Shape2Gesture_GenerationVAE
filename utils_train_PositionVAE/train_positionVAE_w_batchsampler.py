"""
Bottleクラスを席取りLossを用いて多様な方向を生成するためのVAEを学習
しかし、バッチ内に一定の割合(50%)にして学習する必要がある(原因不明)
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset_with_batchsampler import ShapeNetDataset_format_select, BottleQuotaBatchSampler
from model import HandVAE, PartsEncoder_w_TNet, Position_Generater_VAE
from model_pointnet import *
from visualize_method_w_sampler import *
from functions_loss import augmentation_target, sekitori_loss_worst_percent
from functions_pointnet import get_patseg, get_patseg_target
from funtion_else import z_rotation_matrix

"""
席取りLoss(提案手法):
VAEの潜在空間を拘束し、1対多の学習を行うためのLoss関数
席取りLossでLossを確定させ、下位30%を用いる
"""
def sekitori_worst(pred, target, N=12, worst_percent=30):
    target_list = augmentation_target(target, N) # B * N * 23 * 3
    loss_mse, target_idx, loss_mean, idx_min_loss, all_indices, sorted_flat = sekitori_loss_worst_percent(pred, target_list, worst_percent)
    return loss_mse, target_idx, loss_mean, idx_min_loss, all_indices, sorted_flat[0]

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--bottle_label', type=str, default='bo')
    parser.add_argument('--bottle_ratio', type=float, default=0.5)
    parser.add_argument('--drop_last', type=int, default=1)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--sampling', type=int, default=120, help='number of sampling to generate hand from VAE')
    parser.add_argument('--target_num', type=int, default=12, help='number of target to augment for sekitori loss')
    parser.add_argument('--worst_percent', type=int, default=30, help='percent of Loss gradient to use of sekitori loss')

    parser.add_argument('--outf', type=str, default='worst30_sampler', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="dataset", help="dataset path") 
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform") #pointnet option
    parser.add_argument('--select_labels', type=list, default=None, help="what class use of dataset") #["ba", "bo", "ju", "ka", "mu", "pa",  "pc", "po", "va"]

    opt = parser.parse_args()
    print(opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if os.path.exists("Log_tensorboard/"+opt.outf):
        shutil.rmtree("Log_tensorboard/"+opt.outf)
    Writer = SummaryWriter(log_dir="Log_tensorboard/"+opt.outf)

    dataset = ShapeNetDataset_format_select(
        root=opt.dataset,
        data_augmentation=True,
        select_labels=opt.select_labels
        )
    
    train_sampler = BottleQuotaBatchSampler(
        dataset,
        batch_size=opt.batchSize,
        bottle_label=opt.bottle_label,
        bottle_ratio=opt.bottle_ratio,
        drop_last=bool(opt.drop_last),
        seed=opt.manualSeed,
        shuffle_within_batch=True,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=int(opt.workers))
    
    test_dataset = ShapeNetDataset_format_select(
        root=opt.dataset,
        split='val',
        data_augmentation=False,
        select_labels=opt.select_labels
        )
    
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    #学習ではget関数ごとに回転させるため、augmentation(特定の方向に統一されている形状)なしで潜在空間を可視化し評価
    debug_dataset  = ShapeNetDataset_format_select(
        root=opt.dataset,
        data_augmentation=False,
        select_labels=opt.select_labels
        )
    
    debug_dataloader = torch.utils.data.DataLoader(
        debug_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    pointnet = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)
    state_dict_pointnet = torch.load("save_model/pointnet/pointnet_acc_partseg_best.pth", weights_only=True)
    pointnet.load_state_dict(state_dict_pointnet)
    pointnet.eval()

    #学習済みvae
    parts_encoder_l, parts_encoder_r = PartsEncoder_w_TNet(), PartsEncoder_w_TNet()
    state_parts_e_l = torch.load("save_model/pretrained_PartsEncoder_gt/parts_encoder_l_best_total.pth", weights_only=True)
    state_parts_e_r = torch.load("save_model/pretrained_PartsEncoder_gt/parts_encoder_r_best_total.pth", weights_only=True)
    parts_encoder_l.load_state_dict(state_parts_e_l)
    parts_encoder_l.eval()
    parts_encoder_r.load_state_dict(state_parts_e_r)
    parts_encoder_r.eval()

    handvae_r = HandVAE()
    handvae_l = HandVAE()
    state_dict_vae_l = torch.load("save_model/pretrained_HandVAE_formatxy/vae_l_best.pth", weights_only=True)
    state_dict_vae_r = torch.load("save_model/pretrained_HandVAE_formatxy/vae_r_best.pth", weights_only=True)
    handvae_l.load_state_dict(state_dict_vae_l)
    handvae_l.eval()
    handvae_r.load_state_dict(state_dict_vae_r)
    handvae_r.eval()

    #train model 
    position_generater_l, position_generater_r = Position_Generater_VAE(), Position_Generater_VAE()

    optimizer = optim.Adam([{"params":position_generater_l.parameters()},
                            {"params":position_generater_r.parameters()}] ,
                            lr=0.0001, betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    pointnet.cuda()
    handvae_l.cuda()
    handvae_r.cuda()
    parts_encoder_l.cuda()
    parts_encoder_r.cuda()
    position_generater_l.cuda()
    position_generater_r.cuda()

    num_batch = len(dataloader)
    num_sample_per_seat = opt.sampling // opt.target_num #1教師あたりのサンプル数
    
    min_loss_partseg=3
    min_loss_total=100
    min_loss_mse = 1
    min_total_mse_l = 100
    min_total_mse_r = 100
    min_total_mse = 100
    min_total_loss_grab_l, min_total_loss_grab_r = 100, 100

    indices_dict = {"ba":0, "bo":1, "ju":2, "ka":3, "mu":4, "pa":5,  "pc":6, "po":7, "va":8}
    worst_num = int(opt.target_num * opt.worst_percent / 100) #席取りLossで用いる数

    for epoch in range(opt.nepoch):
        train_sampler.set_epoch(epoch)
        #scheduler.step()
        all_sample_zl, all_sample_zr = [], [] # 各サンプルの潜在変数を保存
        all_indices_zl, all_indices_zr = [], [] # 各サンプルのindicesを保存
        flag_b1, flag_b2 = 0, 0 # bottle_0001, bottle_0002のPCA可視化フラグ
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            
            #init
            loss_mse_l_b = 0
            loss_mse_r_b = 0
            kld_l = 0
            kld_r = 0
            loss_l_mse, loss_r_mse = [], []
            loss_mse_l_bottle, loss_mse_r_bottle = 0, 0
            loss_mse_l_else, loss_mse_r_else = 0, 0
            #partseg 
            points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target = data
            batchsize = points.size(0)

            "pl, pr, all_feat, plout, prout = get_patseg(pointnet, points, target)"
            #pl, pr, all_feat, _, _ = get_patseg(pointnet, points, target)
            pl, pe, all_feat, plout, prout = get_patseg_target(pointnet, points, target)
            #手形状生成プロセス
            pf_l, mu_l, logvar_l = parts_encoder_l(pl, all_feat)
            pf_r, mu_r, logvar_r = parts_encoder_r(pr, all_feat)  
            pred_handl = handvae_l.finetune(pf_l)
            pred_handr = handvae_r.finetune(pf_r)
            #手首座標を0,0,0に変換
            wrist_format = torch.tensor([0.5, 0.5, 0.5]).cuda()
            wrist = torch.tensor([0.0, 0.0, 0.0]).cuda().repeat(batchsize, 1, 1)
            pred_handl, pred_handr = pred_handl.view(batchsize, -1, 3) - wrist_format , pred_handr.view(batchsize, -1, 3) - wrist_format
            pred_ges_l, pred_ges_r = torch.cat([wrist, pred_handl], dim=1), torch.cat([wrist, pred_handr], dim=1)
            pred_ges_l_list = torch.tensor([]).cuda() # sampling * 23 * 3
            pred_ges_r_list = torch.tensor([]).cuda()
            #教師の手形状(物体座標系)
            wrist_l_t, wrist_r_t = np.split(wrist_target, 2, axis=1)
            wrist_l_t, wrist_r_t =  wrist_l_t.view(-1, 3).cuda(), wrist_r_t.view(-1, 3).cuda()
            target_hand_l, target_hand_r = np.split(hand_target, 2, axis=1)
            target_hand_l, target_hand_r = target_hand_l.cuda(), target_hand_r.cuda()
            #物体のスケール1に対して手の大きさをスケーリング(教師を用いて)→予測機に変更予定
            hscale_l, hscale_r = np.split(hand_scale, 2, axis=1)
            hscale_l, hscale_r = hscale_l.cuda(), hscale_r.cuda()
            pred_ges_l, pred_ges_r = pred_ges_l / hscale_l, pred_ges_r / hscale_r
            #wrist NN (train) 回転行列と手首座標生成
            #train mode
            position_generater_l, position_generater_r = position_generater_l.train(),  position_generater_r.train() 
            R_l, wrist_l, kld_l, zl = position_generater_l(plout, all_feat, N=opt.sampling)
            R_r, wrist_r, kld_r, zr = position_generater_r(prout, all_feat, N=opt.sampling)
            R_l, R_r = z_rotation_matrix(R_l), z_rotation_matrix(R_r) #sita → R
            #潜在空間zを格納
            all_zl, all_zr  = [], [] 
            all_zl.append(zl)
            all_zr.append(zr)  
            #生成手形状を座標変換
            pred_ges_l_list = pred_ges_l.unsqueeze(1) @ R_l.transpose(2,3) + wrist_l.unsqueeze(2) # B * s * 23 * 3
            pred_ges_r_list = pred_ges_r.unsqueeze(1) @ R_r.transpose(2,3) + wrist_r.unsqueeze(2) # B * s * 23 * 3
            #各教師の席毎のLossを監視するためにretain_grad()→席取りLossの有効性を検証
            pred_ges_l_list.retain_grad()
            pred_ges_r_list.retain_grad()

            #すべてのサンプルのPCAのために配列に格納
            all_sample_zl.append(zl[:,0,:].reshape(batchsize, -1)) 
            all_sample_zr.append(zr[:,0,:].reshape(batchsize, -1))
            #indices格納
            for b in range(batchsize):
                indices = indices_dict[filename[b][:2]]
                all_indices_zl.append(indices)
                all_indices_zr.append(indices)

            #Lossの計算(サンプルごとに)
            for b in range(batchsize):
                pred_l = pred_ges_l_list[b].unsqueeze(0) # 1 * N * 23 * 3
                pred_r = pred_ges_r_list[b].unsqueeze(0) # 1 * N * 23 * 3
                t_l, t_r =  target_hand_l[b].unsqueeze(0), target_hand_r[b].unsqueeze(0) # 1 * 23 * 3
                #ボトルクラスのみ席取りLossでVAEの潜在空間を多様化するように高速(提案手法)
                if "bottle" in filename[b]:
                    #sekitori ボトルのみ lossmse は1サンプルあたり平均mse
                    loss_mse_l, target_idx_l, loss_mean_l, idx_l_min_loss, all_indices_l, worst_histgram_l = sekitori_worst(pred_l, t_l, N=opt.target_num, worst_percent=opt.worst_percent) 
                    loss_mse_r, target_idx_r, loss_mean_r, idx_r_min_loss, all_indices_r, worst_histgram_r = sekitori_worst(pred_r, t_r, N=opt.target_num, worst_percent=opt.worst_percent)
                    loss_mse_l_bottle += loss_mse_l  
                    loss_mse_r_bottle += loss_mse_r  

                    #pca plot
                    # 特定のサンプルのLossを監視
                    if "bottle_0001" == filename[b]:
                        idx = filename.index("bottle_0001")
                        zl1, zr1 = zl[idx], zr[idx]
                        
                        idx_l, idx_r = target_idx_l, target_idx_r # worst
                        minidx_l, minidx_r = idx_l_min_loss, idx_r_min_loss #min 
                        all_loss_l, all_loss_r = all_indices_l, all_indices_r
                        mean_l, mean_r = loss_mean_l, loss_mean_r #hitsgram
                        b1_l, b1_r = all_indices_l[0], all_indices_r[0]

                        #PCA
                        visualize_pca_dual_mean(zl1, idx_l, minidx_l, prefix="zl", modelname=opt.outf, sample_id="0", assigned_indices=all_loss_l, epoch=epoch, count=flag_b1)
                        visualize_pca_dual_mean(zr1, idx_r, minidx_r, prefix="zr", modelname=opt.outf, sample_id="0", assigned_indices=all_loss_r, epoch=epoch, count=flag_b1)
                        #histgram
                        plot_loss_histogram(mean_l, idx_l, prefix="zl", modelname=opt.outf, sample_id="0", epoch=epoch, count=flag_b1)
                        plot_loss_histogram(mean_r, idx_r, prefix="zr", modelname=opt.outf, sample_id="0", epoch=epoch, count=flag_b1)
                        print("save figure")
                        #used gradient histogram
                        wh_l, wh_r = worst_histgram_l, worst_histgram_r
                        plot_sorted_loss_histogram(worst_histgram_l, top_k=worst_num, prefix="zl", modelname=opt.outf, sample_id="0", epoch=epoch, count=flag_b1)
                        plot_sorted_loss_histogram(worst_histgram_r, top_k=worst_num, prefix="zr", modelname=opt.outf, sample_id="0", epoch=epoch, count=flag_b1)
                        flag_b1 += 1

                    # 特定のサンプルのLossを監視 
                    if "bottle_0002" == filename[b]:
                        idx = filename.index("bottle_0002")
                        zl2, zr2 = zl[idx], zr[idx]
                        idx_l, idx_r = target_idx_l, target_idx_r
                        minidx_l, minidx_r = idx_l_min_loss, idx_r_min_loss
                        all_loss_l, all_loss_r = all_indices_l, all_indices_r
                        mean_l, mean_r = loss_mean_l, loss_mean_r
                        visualize_pca_dual_mean(zl2, idx_l, minidx_l, prefix="zl", modelname=opt.outf, sample_id="1", assigned_indices=all_loss_l, epoch=epoch, count=flag_b2)
                        visualize_pca_dual_mean(zr2, idx_r, minidx_r, prefix="zr", modelname=opt.outf, sample_id="1", assigned_indices=all_loss_r, epoch=epoch, count=flag_b2)
                        plot_loss_histogram(mean_l, idx_l, prefix="zl", modelname=opt.outf, sample_id="1", epoch=epoch, count=flag_b2)
                        plot_loss_histogram(mean_r, idx_r, prefix="zr", modelname=opt.outf, sample_id="1", epoch=epoch, count=flag_b2)
                        print("save figure")
                        #used gradient histogram
                        wh_l, wh_r = worst_histgram_l, worst_histgram_r
                        plot_sorted_loss_histogram(worst_histgram_l, top_k=worst_num, prefix="", modelname=opt.outf, sample_id="", epoch=epoch, count=flag_b2)
                        plot_sorted_loss_histogram(worst_histgram_r, top_k=worst_num, prefix="", modelname=opt.outf, sample_id="", epoch=epoch, count=flag_b2)
                        flag_b2 += 1
                        
                else: #その他クラスは通常のMSE Loss 
                    pred_l , pred_r = pred_l[:, :num_sample_per_seat], pred_r[:, :num_sample_per_seat]
                    loss_mse_l = F.mse_loss(pred_l, t_l.unsqueeze(1).repeat(1, num_sample_per_seat, 1, 1), reduction="sum") / num_sample_per_seat
                    loss_mse_r = F.mse_loss(pred_r, t_r.unsqueeze(1).repeat(1, num_sample_per_seat, 1, 1), reduction="sum") / num_sample_per_seat
                    loss_mse_l_else += loss_mse_l 
                    loss_mse_r_else += loss_mse_r 

                #教師1個(69d)の次元あたりのLoss
                print(loss_mse_l / 69, loss_mse_r / 69, filename[b])
            
            loss_mse_l_b = (loss_mse_l_bottle + loss_mse_l_else) / batchsize 
            loss_mse_r_b = (loss_mse_r_bottle + loss_mse_r_else) / batchsize 
            loss = loss_mse_l_b + loss_mse_r_b + (kld_l + kld_r) 
            loss.backward()

            #--- 教師ごとの勾配ノルムを可視化 ---
            os.makedirs(f"grad_histgram/{opt.outf}/bottle", exist_ok=True)
            os.makedirs(f"grad_histgram/{opt.outf}/mug", exist_ok=True)
            os.makedirs(f"grad_histgram/{opt.outf}/pc", exist_ok=True)

            # ---- bottle ----
            if "bottle_0001" in filename:
                idx = filename.index("bottle_0001")
                grad_l_all = pred_ges_l_list.grad[idx]  # 左手
                grad_r_all = pred_ges_r_list.grad[idx]  # 右手
                visualize_grad_bottle(grad_l_all, grad_r_all, b1_l, b1_r, modelname=opt.outf, epoch=epoch)
                
            # ---- mug ----
            if "mug0000" in filename:
                idx = filename.index("mug0000")
                grad_l_all = pred_ges_l_list.grad[idx]  # shape (N=120,23,3)
                grad_r_all = pred_ges_r_list.grad[idx]
                visualize_grad_else(grad_l_all, grad_r_all, class_name="mug", modelname=opt.outf, epoch=epoch)

            # ---- pc ----
            if "pc0000" in filename:
                idx = filename.index("pc0000")
                grad_l_all = pred_ges_l_list.grad[idx]  # shape (N=120,23,3)
                grad_r_all = pred_ges_r_list.grad[idx]
                visualize_grad_else(grad_l_all, grad_r_all, class_name="pc", modelname=opt.outf, epoch=epoch)

            optimizer.step()
            print('[%d: %d/%d] total-train loss: %f' % (epoch, i, num_batch, loss.item()))
            print('[%d: %d/%d] mse_l loss: %f, mse_r loss: %f' % (epoch, i, num_batch, loss_mse_l_b.item(), loss_mse_r_b.item()))
            print('[%d: %d/%d] kld_l: %f, kld_r: %f ' % (epoch, i, num_batch, kld_l.item(), kld_r.item()))
            
            Writer.add_scalars("tensorboad/loss_kld_l",{"train":kld_l.item()},epoch)
            Writer.add_scalars("tensorboad/loss_kld_r",{"train":kld_r.item()},epoch)
            Writer.add_scalars("tensorboad/loss_mse_l",{"train":loss_mse_l_b.item()},epoch)
            Writer.add_scalars("tensorboad/loss_mse_r",{"train":loss_mse_r_b.item()},epoch)
            Writer.add_scalars("tensorboad/loss_mse_l_mean",{"train":loss_mse_l_b.item() / 69},epoch)
            Writer.add_scalars("tensorboad/loss_mse_r_mean",{"train":loss_mse_r_b.item() / 69},epoch)

            if loss < min_loss_total:
                print("min_loss_totalgを更新")
                torch.save(position_generater_l.state_dict(), '%s/position_generater_l_loss_total_best.pth' % (opt.outf))
                torch.save(position_generater_r.state_dict(), '%s/position_generater_r_loss_total_best.pth' % (opt.outf))
                torch.save(optimizer.state_dict(), '%s/optimizer_loss_total_best.pth' % (opt.outf))
                min_loss_total = loss

            if loss_mse_l_b < min_total_mse_l:
                print("min_loss_mse_l更新")
                torch.save(position_generater_l.state_dict(), '%s/position_generater_l_loss_mse_best.pth' % (opt.outf))
                min_total_mse_l = loss_mse_l_b
            
            if loss_mse_r_b < min_total_mse_r:
                print("min_loss_mse_r更新")
                torch.save(position_generater_r.state_dict(), '%s/position_generater_r_loss_mse_best.pth' % (opt.outf))
                min_total_mse_r = loss_mse_r_b
    
        #evalで検証する

        #epochごとにPCA plot
        all_sample_pca(all_sample_zl, all_indices_zl, outf=opt.outf, filename="zl", epoch = epoch)
        all_sample_pca(all_sample_zr, all_indices_zr, outf=opt.outf, filename="zr", epoch = epoch)
        debug_all_zl, debug_all_zr = [], []
        debug_indices = []
        position_generater_l, position_generater_r = position_generater_l.eval(),  position_generater_r.eval()
        
        for data in debug_dataloader:
            points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target = data
            batchsize = points.size(0)
            #_, _, all_feat, _, _ = get_patseg(pointnet, points, target)
            pl, pr, all_feat, plout, prout = get_patseg_target(pointnet, points, target)
            #手形状生成プロセス
            pf_l, mu_l, logvar_l = parts_encoder_l(pl, all_feat)
            pf_r, mu_r, logvar_r = parts_encoder_r(pr, all_feat)  
            pred_handl = handvae_l.finetune(pf_l)
            pred_handr = handvae_r.finetune(pf_r)
            wrist_format = torch.tensor([0.5, 0.5, 0.5]).cuda()
            wrist = torch.tensor([0.0, 0.0, 0.0]).cuda().repeat(batchsize, 1, 1)

            #手首座標を0,0,0に変換
            pred_handl, pred_handr = pred_handl.view(batchsize, -1, 3) - wrist_format , pred_handr.view(batchsize, -1, 3) - wrist_format
            pred_ges_l, pred_ges_r = torch.cat([wrist, pred_handl], dim=1), torch.cat([wrist, pred_handr], dim=1)
            pred_ges_l_list = torch.tensor([]).cuda() # samplingN * 23 * 3
            pred_ges_r_list = torch.tensor([]).cuda()
            #教師手形状(物体座標系)
            wrist_l_t, wrist_r_t = np.split(wrist_target, 2, axis=1)
            wrist_l_t, wrist_r_t =  wrist_l_t.view(-1, 3).cuda(), wrist_r_t.view(-1, 3).cuda()
            target_hand_l, target_hand_r = np.split(hand_target, 2, axis=1)
            target_hand_l, target_hand_r = target_hand_l.cuda(), target_hand_r.cuda()
            
            kld_l, kld_r = 0, 0
            hscale_l, hscale_r = np.split(hand_scale, 2, axis=1)
            hscale_l, hscale_r = hscale_l.cuda(), hscale_r.cuda()
            #潜在変数zを格納
            R_l, wrist_l, kld_Rl, zl = position_generater_l(plout, all_feat, N=1)
            R_r, wrist_r, kld_Rr, zr = position_generater_r(prout, all_feat, N=1)
            debug_all_zl.append(zl[:,0,:].reshape(batchsize, -1))
            debug_all_zr.append(zr[:,0,:].reshape(batchsize, -1))
            for b in range(batchsize):
                indices = indices_dict[filename[b][:2]]
                debug_indices.append(indices)

        #可視化
        all_sample_pca(debug_all_zl, debug_indices, modelname=opt.outf, filename="debugzl", epoch = epoch)
        all_sample_pca(debug_all_zr, debug_indices, modelname=opt.outf, filename="debugzr", epoch = epoch)
        
        os.makedirs('%s/position_generater_l/Epoch' % (opt.outf), exist_ok=True)
        os.makedirs('%s/position_generater_r/Epoch' % (opt.outf), exist_ok=True)
        os.makedirs('%s/optimizer/Epoch' % (opt.outf), exist_ok=True)
        os.makedirs('%s/scheduler/Epoch' % (opt.outf), exist_ok=True)

        if (epoch+1) % 10 == 0: #10エポック毎に保存
            torch.save(position_generater_l.state_dict(), '%s/position_generater_l/Epoch/%s_epoch.pth' % (opt.outf, epoch))
            torch.save(position_generater_r.state_dict(), '%s/position_generater_r/Epoch/%s_epoch.pth' % (opt.outf, epoch))
            torch.save(optimizer.state_dict(), '%s/optimizer/Epoch/%s_epoch.pth' % (opt.outf, epoch))
            torch.save(scheduler.state_dict(), '%s/scheduler/Epoch/%s_epoch.pth' % (opt.outf, epoch))
        scheduler.step()

    Writer.close()
