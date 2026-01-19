from __future__ import print_function
import argparse
import os
import random
import torch

import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset_format_xy import ShapeNetDataset_format_select
from model_pointnet import *
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from model import VAE, PartsEncoder_w_TNet, Position_Generater_VAE
from caclulate_method import *

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils_Hand_Generation.visualize_method import *
matplotlib.use("Agg")

def sekitori(pred, target):
    target_list = rotate_targets_z(target, N = 12) # B * N * 23 * 3
    loss_mse, target_idx, loss_mean, idx_min_loss, all_indices = sekitori_loss_sum(pred, target_list)
    return loss_mse, target_idx, loss_mean, idx_min_loss, all_indices

    
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers',)
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--sampling', type=int, default=120, help='number of sampling to generate hand from VAE')
    parser.add_argument('--topK', type=int, default=10, help='persent of topK to use backward')

    parser.add_argument('--outf', type=str, default='sekitori_AAA', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="neuralnet_dataset_unity", help="dataset path") # dataset2, neuralnet_dataset_unity dataset_3class
    parser.add_argument('--class_choice', type=str, default='sotuken', help="class_choice")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform") #pointnetの設定
    parser.add_argument('--select_labels', type=list, default=None, help="what class use of dataset") #["ju", "mu", "bo", "pc"]
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
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
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

    num_classes = dataset.num_seg_classes
    print('classes', num_classes)

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    pointnet_classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
    state_dict_pointnet = torch.load("save_pretrained_partsseg/pointnet_model_sotuken_acc_partseg_best.pth", weights_only=True)
    pointnet_classifier.load_state_dict(state_dict_pointnet)
    pointnet_classifier.eval()
    #train model 
    "rotation matrix NN"
    #sita_generater_l, sita_generater_r = QuaterionVAE5(), QuaterionVAE5()
    sita_generater_l, sita_generater_r = Position_Generater_VAE(), Position_Generater_VAE()
    # state_sita_l = torch.load("test_targetAAA/sita_generater_l_loss_mse_best.pth", weights_only=True) #train922/sita_generater_l/Epoch/49_epoch.pth, train922/sita_generater_l_loss_mse_best.pth
    # state_sita_r = torch.load("test_targetAAA/sita_generater_r_loss_mse_best.pth", weights_only=True) #train922/sita_generater_r/Epoch/49_epoch.pth, train922/sita_generater_r_loss_mse_best.pth
    # sita_generater_l.load_state_dict(state_sita_l)
    # sita_generater_r.load_state_dict(state_sita_r)
    # sita_generater_l.eval(), sita_generater_r.eval()

    #学習済みvae
    "initial hand vae"

    parts_encoder_classifier_l, parts_encoder_classifier_r = PartsEncoder_w_TNet(), PartsEncoder_w_TNet()
    # state_parts_e_l = torch.load("fps_format2_target/parts_encoder_l_model_sotuken_loss_mse_best.pth", weights_only=True)
    # state_parts_e_r = torch.load("fps_format2_target/parts_encoder_r_model_sotuken_loss_mse_best.pth", weights_only=True)
    state_parts_e_l = torch.load("fps_formatxy2/parts_encoder_l_model_sotuken_loss_mse_best.pth", weights_only=True)
    state_parts_e_r = torch.load("fps_formatxy2/parts_encoder_r_model_sotuken_loss_mse_best.pth", weights_only=True)

    parts_encoder_classifier_l.load_state_dict(state_parts_e_l)
    parts_encoder_classifier_l.eval()
    parts_encoder_classifier_r.load_state_dict(state_parts_e_r)
    parts_encoder_classifier_r.eval()

    vae_classifier_r = VAE()
    vae_classifier_l = VAE()
    state_dict_vae_l = torch.load("save_model/save_pretrained_VAE_formatxy/vae_l_sotuken_loss_total_best.pth", weights_only=True)
    state_dict_vae_r = torch.load("save_model/save_pretrained_VAE_formatxy/vae_r_sotuken_loss_total_best.pth", weights_only=True)
    vae_classifier_l.load_state_dict(state_dict_vae_l)
    vae_classifier_l.eval()
    vae_classifier_r.load_state_dict(state_dict_vae_r)
    vae_classifier_r.eval()

    optimizer = optim.Adam([{"params":sita_generater_l.parameters()},
                            {"params":sita_generater_r.parameters()}] ,
                            lr=0.0001, betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    pointnet_classifier.cuda()
    vae_classifier_l.cuda()
    vae_classifier_r.cuda()
    parts_encoder_classifier_l.cuda()
    parts_encoder_classifier_r.cuda()
    sita_generater_l.cuda()
    sita_generater_r.cuda()

    num_batch = len(dataset) / opt.batchSize
    best_partseg_acc=0
    
    min_loss_partseg=3
    min_loss_total=100
    min_loss_mse = 1
    min_total_mse_l = 100
    min_total_mse_r = 100

    min_total_mse = 100
    min_total_loss_grab_l, min_total_loss_grab_r = 100, 100

    topK = int(opt.sampling * opt.topK *0.01)

    indices_dict = {"ba":0, "bo":1, "ju":2, "ka":3, "mu":4, "pa":5,  "pc":6, "po":7, "va":8}

    for epoch in range(opt.nepoch):
        #scheduler.step()
        all_sample_zl, all_sample_zr = [], [] # 各サンプルの潜在変数を保存
        all_indices_zl, all_indices_zr = [], [] # 各サンプルの潜在変数のindicesを保存
        for i, data in enumerate(dataloader, 0):

            optimizer.zero_grad()
            #train mode
            sita_generater_l, sita_generater_r = sita_generater_l.train(),  sita_generater_r.train() 
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

            "pl, pr, all_feat, plout, prout = get_patseg(pointnet_classifier, points, target)"
            _, _, all_feat, _, _ = get_patseg(pointnet_classifier, points, target)
            pl, pr, _, plout, prout = get_patseg_target(pointnet_classifier, points, target)
            #パーツの特徴ベクトル取得
            pf_l, mu_l, logvar_l = parts_encoder_classifier_l(pl, all_feat)
            pf_r, mu_r, logvar_r = parts_encoder_classifier_r(pr, all_feat)  
            #基準の手を生成
            pred_handl = vae_classifier_l.finetune(pf_l)
            pred_handr = vae_classifier_r.finetune(pf_r)
            wrist_format = torch.tensor([0.5, 0.5, 0.5]).cuda()
            wrist = torch.tensor([0.0, 0.0, 0.0]).cuda().repeat(batchsize, 1, 1)
            #手首座標を0,0,0に変換
            pred_handl, pred_handr = pred_handl.view(batchsize, -1, 3) - wrist_format , pred_handr.view(batchsize, -1, 3) - wrist_format

            pred_ges_l, pred_ges_r = torch.cat([wrist, pred_handl], dim=1), torch.cat([wrist, pred_handr], dim=1)
            pred_ges_l_list = torch.tensor([]).cuda() # sampling * 23 * 3
            pred_ges_r_list = torch.tensor([]).cuda()
            #教師手首
            wrist_l_t, wrist_r_t = np.split(wrist_target, 2, axis=1)
            wrist_l_t, wrist_r_t =  wrist_l_t.view(-1, 3).cuda(), wrist_r_t.view(-1, 3).cuda()
            target_hand_l, target_hand_r = np.split(hand_target, 2, axis=1)
            target_hand_l, target_hand_r = target_hand_l.cuda(), target_hand_r.cuda()
            
            #同じ入力でN回生成する。
            all_zl, all_zr  = [], [] 
            hscale_l, hscale_r = np.split(hand_scale, 2, axis=1)
            hscale_l, hscale_r = hscale_l.cuda(), hscale_r.cuda()

            #wrist NN (train) 回転行列と手首座標生成
            pred_ges_l_list, pred_ges_r_list = torch.tensor([]), torch.tensor([])

            R_l, wrist_l, kld_Rl, zl = sita_generater_l(plout, all_feat, N=opt.sampling)
            R_r, wrist_r, kld_Rr, zr = sita_generater_r(prout, all_feat, N=opt.sampling)
            
            kld_l = kld_Rl
            kld_r = kld_Rr

            R_l, R_r = z_rotation_matrix(R_l), z_rotation_matrix(R_r) #[2, 10, 3, 3]
            all_zl.append(zl)
            all_zr.append(zr)  
            pred_ges_l, pred_ges_r = pred_ges_l / hscale_l, pred_ges_r / hscale_r
            
            pred_ges_l_list = pred_ges_l.unsqueeze(1) @ R_l.transpose(2,3) + wrist_l.unsqueeze(2) # B * s * 23 * 3
            pred_ges_r_list = pred_ges_r.unsqueeze(1) @ R_r.transpose(2,3) + wrist_r.unsqueeze(2) # B * s * 23 * 3
            
            pred_ges_l_list.retain_grad()
            pred_ges_r_list.retain_grad()

            #すべてのサンプルのPCAのために配列に格納
            all_sample_zl.append(zl[:,:,:2].reshape(batchsize * 2, -1)) 
            all_sample_zr.append(zr[:,:,:2].reshape(batchsize * 2, -1))

            for b in range(batchsize):
                indices = indices_dict[filename[b][:2]]
                for n in range(2):
                    all_indices_zl.append(indices)
                    all_indices_zr.append(indices)


            #losses_worst, worst_idx, loss_mean_target, indices_min_loss, all_indices
            "losses_worst:勾配計算で使うLoss, worst_idx:どの教師idxがその最大値をとっているのか？, "
            "loss_mean_target:各教師のLossの平均, indices_min_loss:そもそもどの教師と一番近いか, "
            "all_indices:教師ごとのサンプルidx"
            #Lossの計算
            for b in range(batchsize):
                pred_l = pred_ges_l_list[b].unsqueeze(0) # 1 * N * 23 * 3
                pred_r = pred_ges_r_list[b].unsqueeze(0) # 1 * N * 23 * 3
                t_l, t_r =  target_hand_l[b].unsqueeze(0), target_hand_r[b].unsqueeze(0) # 1 * 23 * 3
                
                if "bottle" in filename[b]:
                    #sekitori ボトルのみ lossmse は1サンプルあたりの平均mse
                    loss_mse_l, target_idx_l, loss_mean_l, idx_l_min_loss, all_indices_l = sekitori(pred_l, t_l) 
                    loss_mse_r, target_idx_r, loss_mean_r, idx_r_min_loss, all_indices_r = sekitori(pred_r, t_r)

                    loss_mse_l_bottle += loss_mse_l  
                    loss_mse_r_bottle += loss_mse_r  

                    #pca plot
                    # 特定のサンプルの遷移を監視
                    if "bottle_0001" == filename[b]:
                        idx = filename.index("bottle_0001")
                        zl1, zr1 = zl[idx], zr[idx]
                        
                        idx_l, idx_r = target_idx_l, target_idx_r # worst
                        minidx_l, minidx_r = idx_l_min_loss, idx_r_min_loss #min 
                        all_loss_l, all_loss_r = all_indices_l, all_indices_r
                        mean_l, mean_r = loss_mean_l, loss_mean_r #hitsgram
                        b1_l, b1_r = all_indices_l[0], all_indices_r[0]

                        #PCA
                        visualize_pca_dual_mean(zl1, idx_l, minidx_l, prefix="zl", modelname=opt.outf, sample_id="0", assigned_indices=all_loss_l, epoch=epoch)
                        visualize_pca_dual_mean(zr1, idx_r, minidx_r, prefix="zr", modelname=opt.outf, sample_id="0", assigned_indices=all_loss_r, epoch=epoch)
                        #histgram
                        plot_loss_histogram(mean_l, idx_l, prefix="zl", modelname=opt.outf, sample_id="0", epoch=epoch)
                        plot_loss_histogram(mean_r, idx_r, prefix="zr", modelname=opt.outf, sample_id="0", epoch=epoch)
                        print("save figure")

                        #worst hisgram
                        # wh_l, wh_r = worst_histgram_l, worst_histgram_r
                        # plot_sorted_loss_histogram(worst_histgram_l, top_k=36, prefix="zl", modelname=opt.outf, sample_id="0", epoch=epoch)
                        # plot_sorted_loss_histogram(worst_histgram_r, top_k=36, prefix="zr", modelname=opt.outf, sample_id="0", epoch=epoch)
                        
                    
                    if "bottle_0002" == filename[b]:
                        idx = filename.index("bottle_0002")
                        zl2, zr2 = zl[idx], zr[idx]
                        idx_l, idx_r = target_idx_l, target_idx_r
                        minidx_l, minidx_r = idx_l_min_loss, idx_r_min_loss
                        all_loss_l, all_loss_r = all_indices_l, all_indices_r
                        mean_l, mean_r = loss_mean_l, loss_mean_r
                        visualize_pca_dual_mean(zl2, idx_l, minidx_l, prefix="zl", modelname=opt.outf, sample_id="1", assigned_indices=all_loss_l, epoch=epoch)
                        visualize_pca_dual_mean(zr2, idx_r, minidx_r, prefix="zr", modelname=opt.outf, sample_id="1", assigned_indices=all_loss_r, epoch=epoch)
                        plot_loss_histogram(mean_l, idx_l, prefix="zl", modelname=opt.outf, sample_id="1", epoch=epoch)
                        plot_loss_histogram(mean_r, idx_r, prefix="zr", modelname=opt.outf, sample_id="1", epoch=epoch)
                        print("save figure")

                        #worst hisgram
                        # wh_l, wh_r = worst_histgram_l, worst_histgram_r
                        # plot_sorted_loss_histogram(worst_histgram_l, top_k=36, prefix="", modelname="", sample_id="", epoch=epoch)
                        # plot_sorted_loss_histogram(worst_histgram_r, top_k=36, prefix="", modelname="", sample_id="", epoch=epoch)
                        
                
                else: 
                    #MSE
                    # loss_mse_l = F.mse_loss(pred_l, t_l.unsqueeze(1).repeat(1, 120, 1, 1)) /120
                    # loss_mse_r = F.mse_loss(pred_r, t_r.unsqueeze(1).repeat(1, 120, 1, 1)) /120
                    #関取の教師1個に対してpredは10こで計算するから
                    pred_l , pred_r = pred_l[:, :30], pred_r[:, :30]
                    loss_mse_l = F.mse_loss(pred_l, t_l.unsqueeze(1).repeat(1, 30, 1, 1), reduction="sum") /30
                    loss_mse_r = F.mse_loss(pred_r, t_r.unsqueeze(1).repeat(1, 30, 1, 1), reduction="sum") /30
                    # 1サンプルごとのMSE誤差(69dの和)
                    loss_mse_l_else += loss_mse_l 
                    loss_mse_r_else += loss_mse_r 
                #xyzそれぞれどれだけ異なるか
                #教師1個あたり
                print(loss_mse_l / 69, loss_mse_r / 69, filename[b])
                
            loss_mse_l_b = (loss_mse_l_bottle + loss_mse_l_else) / batchsize #ボトルのみ
            loss_mse_r_b = (loss_mse_r_bottle + loss_mse_r_else) / batchsize #その他　8クラス

            # mse_bottle = loss_mse_l_bottle + loss_mse_r_bottle 
            # mse_else = loss_mse_l_else + loss_mse_r_else 

            loss = loss_mse_l_b + loss_mse_r_b + (kld_l + kld_r) 
            loss.backward()



            # 保存フォルダの作成
            os.makedirs(f"grad_histgram/{opt.outf}/bottle", exist_ok=True)
            os.makedirs(f"grad_histgram/{opt.outf}/mug", exist_ok=True)
            os.makedirs(f"grad_histgram/{opt.outf}/pc", exist_ok=True)

            # --- 例: filename, pred_ges_l_list, pred_ges_r_list, b1_l, b2_r, epoch が既に定義されている前提 ---

            if "bottle_0001" in filename:
                b = filename.index("bottle_0001")

                # pred の grad（形状: [N=120, 23, 3]）
                grad_l_all = pred_ges_l_list.grad[b]  # 左手
                grad_r_all = pred_ges_r_list.grad[b]  # 右手

                # 割り当てインデックス
                T, K = b1_l.shape

                teacher_grad_l = [[] for _ in range(T)]
                teacher_grad_r = [[] for _ in range(T)]

                for t in range(T):
                    for k in range(K):
                        pred_idx_l = int(b1_l[t, k]) #せきとりのidxを参照。各教師に結びついたpredのidxが保管されている。 
                        pred_idx_r = int(b1_r[t, k])

                        grad_l = grad_l_all[pred_idx_l]  # shape (23,3)
                        grad_r = grad_r_all[pred_idx_r]

                        teacher_grad_l[t].append(grad_l.detach().cpu())
                        teacher_grad_r[t].append(grad_r.detach().cpu())

                # Tensor化
                for t in range(T):
                    teacher_grad_l[t] = torch.stack(teacher_grad_l[t], dim=0)  # (K,23,3)
                    teacher_grad_r[t] = torch.stack(teacher_grad_r[t], dim=0)

                # 教師ごとの L2 ノルム
                grad_l_stats = torch.stack(teacher_grad_l, dim=0)
                print(grad_l_stats.size())
                grad_l_stats = torch.norm(torch.stack(teacher_grad_l, dim=0), dim=(2,3)).mean(dim=1).numpy()
                grad_r_stats = torch.norm(torch.stack(teacher_grad_r, dim=0), dim=(2,3)).mean(dim=1).numpy()

                # 左手
                plt.figure(figsize=(8,5))
                plt.bar(range(T), grad_l_stats, alpha=0.8)
                plt.xlabel("Teacher index")
                plt.ylabel("L2 Norm of Gradients (Left hand)")
                plt.title("Per-teacher Gradient L2 Norm (Left Hand)")
                plt.tight_layout()
                plt.savefig(f"grad_histgram/{opt.outf}/bottle/grad_norms_left_{epoch}.png")
                plt.close()

                # 右手
                plt.figure(figsize=(8,5))
                plt.bar(range(T), grad_r_stats, alpha=0.8)
                plt.xlabel("Teacher index")
                plt.ylabel("L2 Norm of Gradients (Right hand)")
                plt.title("Per-teacher Gradient L2 Norm (Right Hand)")
                plt.tight_layout()
                plt.savefig(f"grad_histgram/{opt.outf}/bottle/grad_norms_right_{epoch}.png")
                plt.close()


            # ---- mug ----
            if "mug0000" in filename:
                b = filename.index("mug0000")

                grad_l_all = pred_ges_l_list.grad[b]  # shape (N=120,23,3)
                grad_r_all = pred_ges_r_list.grad[b]

                # 1次元化して L2 ノルム
                grad_l_mean = torch.norm(grad_l_all, dim=(1,2)).mean().detach().cpu().numpy()
                grad_r_mean = torch.norm(grad_r_all, dim=(1,2)).mean().detach().cpu().numpy()

                # 左手
                plt.figure(figsize=(6,6))
                plt.bar([0], [grad_l_mean], alpha=0.8)
                plt.xticks([0], ["Mug"])
                plt.ylabel("L2 Norm of Gradients (Left hand)")
                plt.title("Left Hand Gradient L2 Norm")
                plt.tight_layout()
                plt.savefig(f"grad_histgram/{opt.outf}/mug/grad_norm_left_{epoch}.png")
                plt.close()

                # 右手
                plt.figure(figsize=(6,6))
                plt.bar([0], [grad_r_mean], alpha=0.8)
                plt.xticks([0], ["Mug"])
                plt.ylabel("L2 Norm of Gradients (Right hand)")
                plt.title("Right Hand Gradient L2 Norm")
                plt.tight_layout()
                plt.savefig(f"grad_histgram/{opt.outf}/mug/grad_norm_right_{epoch}.png")
                plt.close()


            # ---- pc ----
            if "pc0000" in filename:
                b = filename.index("pc0000")

                grad_l_all = pred_ges_l_list.grad[b]  # shape (N=120,23,3)
                grad_r_all = pred_ges_r_list.grad[b]

                grad_l_mean = torch.norm(grad_l_all, dim=(1,2)).mean().detach().cpu().numpy()
                grad_r_mean = torch.norm(grad_r_all, dim=(1,2)).mean().detach().cpu().numpy()

                # 左手
                plt.figure(figsize=(6,6))
                plt.bar([0], [grad_l_mean], alpha=0.8)
                plt.xticks([0], ["PC"])
                plt.ylabel("L2 Norm of Gradients (Left hand)")
                plt.title("Left Hand Gradient L2 Norm")
                plt.tight_layout()
                plt.savefig(f"grad_histgram/{opt.outf}/pc/grad_norm_left_{epoch}.png")
                plt.close()

                # 右手
                plt.figure(figsize=(6,6))
                plt.bar([0], [grad_r_mean], alpha=0.8)
                plt.xticks([0], ["PC"])
                plt.ylabel("L2 Norm of Gradients (Right hand)")
                plt.title("Right Hand Gradient L2 Norm")
                plt.tight_layout()
                plt.savefig(f"grad_histgram/{opt.outf}/pc/grad_norm_right_{epoch}.png")
                plt.close()


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
                torch.save(sita_generater_l.state_dict(), '%s/sita_generater_l_loss_total_best.pth' % (opt.outf))
                torch.save(sita_generater_r.state_dict(), '%s/sita_generater_r_loss_total_best.pth' % (opt.outf))
                torch.save(optimizer.state_dict(), '%s/optimizer_loss_total_best.pth' % (opt.outf))
                min_loss_total = loss

            if loss_mse_l_b < min_total_mse_l:
                print("min_loss_mse_l更新")
                torch.save(sita_generater_l.state_dict(), '%s/sita_generater_l_loss_mse_best.pth' % (opt.outf))
                #torch.save(wrist_generater_l.state_dict(), '%s/wrist_generater_l_%s_loss_total_best.pth' % (opt.outf, opt.class_choice))
                min_total_mse_l = loss_mse_l_b
            
            if loss_mse_r_b < min_total_mse_r:
                print("min_loss_mse_r更新")
                torch.save(sita_generater_r.state_dict(), '%s/sita_generater_r_loss_mse_best.pth' % (opt.outf))
                #torch.save(wrist_generater_r.state_dict(), '%s/wrist_generater_r_%s_loss_total_best.pth' % (opt.outf, opt.class_choice))
                min_total_mse_r = loss_mse_r_b
    
        #evalで検証する





        #epochごとにPCA plot
        all_sample_pca(all_sample_zl, all_indices_zl, outf=opt.outf, filename="zl", epoch = epoch)
        all_sample_pca(all_sample_zr, all_indices_zr, outf=opt.outf, filename="zr", epoch = epoch)

        debug_all_zl, debug_all_zr = [], []
        debug_indices = []
        sita_generater_l, sita_generater_r = sita_generater_l.eval(),  sita_generater_r.eval()
        for data in debug_dataloader:
            points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target = data
            batchsize = points.size(0)
            "pl, pr, all_feat, plout, prout = get_patseg(pointnet_classifier, points, target)"
            _, _, all_feat, _, _ = get_patseg(pointnet_classifier, points, target)
            pl, pr, _, plout, prout = get_patseg_target(pointnet_classifier, points, target)
            #パーツの特徴ベクトル取得
            pf_l, mu_l, logvar_l = parts_encoder_classifier_l(pl, all_feat)
            pf_r, mu_r, logvar_r = parts_encoder_classifier_r(pr, all_feat)  
            #基準の手を生成
            pred_handl = vae_classifier_l.finetune(pf_l)
            pred_handr = vae_classifier_r.finetune(pf_r)
            wrist_format = torch.tensor([0.5, 0.5, 0.5]).cuda()
            wrist = torch.tensor([0.0, 0.0, 0.0]).cuda().repeat(batchsize, 1, 1)
            #手首座標を0,0,0に変換
            pred_handl, pred_handr = pred_handl.view(batchsize, -1, 3) - wrist_format , pred_handr.view(batchsize, -1, 3) - wrist_format

            pred_ges_l, pred_ges_r = torch.cat([wrist, pred_handl], dim=1), torch.cat([wrist, pred_handr], dim=1)
            pred_ges_l_list = torch.tensor([]).cuda() # sampling * 23 * 3
            pred_ges_r_list = torch.tensor([]).cuda()
            #教師手首
            wrist_l_t, wrist_r_t = np.split(wrist_target, 2, axis=1)
            wrist_l_t, wrist_r_t =  wrist_l_t.view(-1, 3).cuda(), wrist_r_t.view(-1, 3).cuda()
            target_hand_l, target_hand_r = np.split(hand_target, 2, axis=1)
            target_hand_l, target_hand_r = target_hand_l.cuda(), target_hand_r.cuda()
            
            #同じ入力でN回生成する。
            kld_l, kld_r = 0, 0
            hscale_l, hscale_r = np.split(hand_scale, 2, axis=1)
            hscale_l, hscale_r = hscale_l.cuda(), hscale_r.cuda()

            #wrist NN (train) 回転行列と手首座標生成

            R_l, wrist_l, kld_Rl, zl = sita_generater_l(plout, all_feat, N=1)
            R_r, wrist_r, kld_Rr, zr = sita_generater_r(prout, all_feat, N=1)
            
            #すべてのサンプルのPCAのために配列に格納
            #debug_all_zl.append(zl[:,:,:2].reshape(batchsize * 2, -1)) 
            debug_all_zl.append(zl[:,:,:2].reshape(batchsize, -1))
            debug_all_zr.append(zr[:,:,:2].reshape(batchsize, -1))
            #debug_all_zr.append(zr[:,:,:2].reshape(batchsize , -1))

            for b in range(batchsize):
                indices = indices_dict[filename[b][:2]]
                debug_indices.append(indices)

        all_sample_pca(debug_all_zl, debug_indices, outf=opt.outf, filename="debugzl", epoch = epoch)
        all_sample_pca(debug_all_zr, debug_indices, outf=opt.outf, filename="debugzr", epoch = epoch)
        
        os.makedirs('%s/sita_generater_l/Epoch' % (opt.outf), exist_ok=True)
        os.makedirs('%s/sita_generater_r/Epoch' % (opt.outf), exist_ok=True)
        os.makedirs('%s/optimizer/Epoch' % (opt.outf), exist_ok=True)
        os.makedirs('%s/scheduler/Epoch' % (opt.outf), exist_ok=True)

        if (epoch+1) % 4 == 0: #5epochずつ保存
            torch.save(sita_generater_l.state_dict(), '%s/sita_generater_l/Epoch/%s_epoch.pth' % (opt.outf, epoch))
            torch.save(sita_generater_r.state_dict(), '%s/sita_generater_r/Epoch/%s_epoch.pth' % (opt.outf, epoch))
            torch.save(optimizer.state_dict(), '%s/optimizer/Epoch/%s_epoch.pth' % (opt.outf, epoch))
            torch.save(scheduler.state_dict(), '%s/scheduler/Epoch/%s_epoch.pth' % (opt.outf, epoch))
        scheduler.step()

    Writer.close()

    print("----------")
    print("学習終了")

   
    

   