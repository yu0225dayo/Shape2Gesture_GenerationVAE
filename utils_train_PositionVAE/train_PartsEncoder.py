"""
Parts → Hand(format)
事前学習した手形状生成VAEを用いてEncoderを学習
学習したPointNetを用いて学習
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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset_format_xy import ShapeNetDataset_format
from model import HandVAE, PartsEncoder_w_TNet
from model_pointnet import *
from visualize_method import *
from caclulate_method import *

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--sampling', type=int, default=5, help='number of sampling to train for')
    
    parser.add_argument('--outf', type=str, default='pretrained_PartsEncoder', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="dataset", help="dataset path")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
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

    dataset = ShapeNetDataset_format(
        root=opt.dataset,
        data_augmentation=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = ShapeNetDataset_format(
        root=opt.dataset,
        split='val',
        data_augmentation=False)
    
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    print(dataset)
    print(len(dataset), len(test_dataset))

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    #pretrained model
    pointnet = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)
    state_dict_pointnet = torch.load("save_model/pointnet/pointnet_acc_partseg_best.pth", weights_only=True)
    pointnet.load_state_dict(state_dict_pointnet)
    pointnet.eval()

    vae_l = HandVAE()
    vae_r = HandVAE()
    state_dict_vae_l = torch.load("save_model/pretrained_HandVAE_formatxy/vae_l_best.pth", weights_only=True)
    state_dict_vae_r = torch.load("save_model/pretrained_HandVAE_formatxy/vae_r_best.pth", weights_only=True)
    vae_l.load_state_dict(state_dict_vae_l)
    vae_l.eval()
    vae_r.load_state_dict(state_dict_vae_r)
    vae_r.eval()

    #train model 
    parts_encoder_l, parts_encoder_r = PartsEncoder_w_TNet(), PartsEncoder_w_TNet()
    optimizer = optim.Adam([{"params":parts_encoder_l.parameters()},
                            {"params":parts_encoder_r.parameters()}],
                            lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    pointnet.cuda()
    vae_l.cuda()
    vae_r.cuda()
    parts_encoder_l.cuda()
    parts_encoder_r.cuda()

    #一応vaeのgradをfalseに→optimizerになければいいはず
    for param in vae_l.decoder.parameters():
        param.requires_grad = False
    for param in vae_r.decoder.parameters():
        param.requires_grad = False

    num_batch = len(dataset) / opt.batchSize
    best_partseg_acc=0

    min_loss_total=100
    min_loss_mse = 1
    min_total_mse_l = 10
    min_total_mse_r = 10

    min_total_mse = 100

    for epoch in range(opt.nepoch):
        #scheduler.step()
        for i, data in enumerate(dataloader, 0):
            #print(i, data)
            optimizer.zero_grad()
            points, target, input_hands, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
            batch_size = points.size(0)
            points, target = points.cuda(), target.cuda() 
            #手首を(.5,.5,.5)に 
            hand=np.split(hand_format, 2, axis=1)            
            hand_l = hand[0]
            hand_r = hand[1]
            for k in range(batch_size):
                hand_l[k] = hand_l[k] - hand_l[k][0] + torch.tensor([0.5, 0.5, 0.5])
                hand_r[k] = hand_r[k] - hand_r[k][0] + torch.tensor([0.5, 0.5, 0.5])
            hand_l = hand_l.reshape(batch_size,69)
            hand_r = hand_r.reshape(batch_size,69)
            #target
            hand_l_gt = hand_l[:,3:]
            hand_r_gt = hand_r[:,3:]
            hand_l_gt, hand_r_gt,  all_feat = hand_l_gt.cuda(), hand_r_gt.cuda(), all_feat.cuda()
            #train()
            parts_encoder_l = parts_encoder_l.train() 
            parts_encoder_r = parts_encoder_r.train()
            #init
            loss_mse_l = 0
            loss_mse_r = 0
            loss_latent_l = 0
            loss_latent_r = 0
            kld_l = 0
            kld_r = 0
            loss_mse_sita_l = 0
            loss_mse_sita_r = 0
            #sampling 
            for t in range(opt.sampling):
                #pointnet parts segmentation
                pl, pr, all_feat, _, _ = get_patseg(pointnet, points, target)
                pf_l, mu_l, logvar_l = parts_encoder_l(pl, all_feat)
                pf_r, mu_r, logvar_r = parts_encoder_r(pr, all_feat)
                kld_l += torch.mean(-0.5 * torch.sum(1 + logvar_l - mu_l ** 2 - logvar_l.exp(), dim = 1), dim = 0)
                kld_r += torch.mean(-0.5 * torch.sum(1 + logvar_r - mu_r ** 2 - logvar_r.exp(), dim = 1), dim = 0)

                pred_handl = vae_l.finetune(pf_l)
                pred_handr = vae_r.finetune(pf_r)

                loss_mse_l += F.mse_loss(pred_handl, hand_l_gt ,reduction="sum") / batch_size
                loss_mse_r += F.mse_loss(pred_handr, hand_r_gt ,reduction="sum") / batch_size

            loss = loss_mse_l + loss_mse_r + kld_l + kld_r 
            loss = loss / opt.sampling
            loss_mse_l, loss_mse_r = loss_mse_l / opt.sampling, loss_mse_r/ opt.sampling
            kld_l, kld_r = kld_l/ opt.sampling, kld_r/ opt.sampling

            loss.backward()
            optimizer.step()

            print('[%d: %d/%d] total-train loss: %f' % (epoch, i, num_batch, loss.item()))
            print('[%d: %d/%d] mse_l loss: %f, mse_r loss: %f' % (epoch, i, num_batch, loss_mse_l.item(),  loss_mse_r.item()))
            print('[%d: %d/%d] kld_l: %f, kld_r: %f ' % (epoch, i, num_batch, kld_l.item(), kld_r.item()))
            
            #loss
            Writer.add_scalars("tensorboad/loss_mse_l",{"train":loss_mse_l.item()},epoch)
            Writer.add_scalars("tensorboad/loss_mse_r",{"train":loss_mse_r.item()},epoch)
            Writer.add_scalars("tensorboad/loss_kld_l",{"train":kld_l.item()},epoch)
            Writer.add_scalars("tensorboad/loss_kld_r",{"train":kld_r.item()},epoch)

            #eval()
            if (i+1) % 9 == 0:
                j, data = next(enumerate(testdataloader,0))
                points, target, input_hands, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
                batch_size = points.size(0)
                points, target = points.cuda(), target.cuda()  
                #手首を(.5,.5,.5)に
                hand = np.split(hand_format,2,axis=1)  
                hand_l = hand[0]
                hand_r = hand[1]
                for k in range(batch_size):
                    hand_l[k] = hand_l[k] - hand_l[k][0] + torch.tensor([0.5, 0.5, 0.5])
                    hand_r[k] = hand_r[k] - hand_r[k][0] + torch.tensor([0.5, 0.5, 0.5])
                hand_l = hand_l.reshape(batch_size,69)
                hand_r = hand_r.reshape(batch_size,69)
                #target
                hand_l_gt = hand_l[:, 3:]
                hand_r_gt = hand_r[:, 3:]
                hand_l_gt, hand_r_gt = hand_l_gt.cuda(), hand_r_gt.cuda()

                #eval()
                parts_encoder_l = parts_encoder_l.eval()
                parts_encoder_r = parts_encoder_r.eval() 

                #pointnet part segmentation
                pl, pr, all_feat, _, _ = get_patseg(pointnet, points, target)
                pf_l, mu_l, logvar_l = parts_encoder_l(pl, all_feat)
                pf_r, mu_r, logvar_r = parts_encoder_r(pr, all_feat)

                kld_l = torch.mean(-0.5 * torch.sum(1 + logvar_l - mu_l ** 2 - logvar_l.exp(), dim = 1), dim = 0)
                kld_r = torch.mean(-0.5 * torch.sum(1 + logvar_r - mu_r ** 2 - logvar_r.exp(), dim = 1), dim = 0)
            
                pred_handl = vae_l.finetune(pf_l)
                pred_handr = vae_r.finetune(pf_r)

                loss_mse_l = F.mse_loss(pred_handl, hand_l_gt, reduction="sum") / batch_size
                loss_mse_r = F.mse_loss(pred_handr, hand_r_gt, reduction="sum") / batch_size

                loss = loss_mse_l + loss_mse_r + kld_l + kld_r 

                print('[%d: %d/%d] %s total-test loss: %f' % (epoch, i, num_batch, blue("test"), loss.item()))
                print('[%d: %d/%d] %s mse_l loss: %f mse_r loss: %f' % (epoch, i, num_batch,blue("test"), loss_mse_l.item(),  loss_mse_r.item())) 
                print('[%d: %d/%d] %s kld_l: %f, kld_r: %f' % (epoch, i, num_batch, blue("test"), kld_l.item(), kld_r.item())) 
                
                #loss
                Writer.add_scalars("tensorboad/loss_mse_l",{"val":loss_mse_l.item()},epoch)
                Writer.add_scalars("tensorboad/loss_mse_r",{"val":loss_mse_r.item()},epoch)
                Writer.add_scalars("tensorboad/loss_kld_l",{"val":kld_l.item()},epoch)
                Writer.add_scalars("tensorboad/loss_kld_r",{"val":kld_r.item()},epoch)

                #model save 
                #total
                if loss < min_loss_total:
                    print("----------")
                    print("min_loss_totalgを更新")
                    torch.save(parts_encoder_l.state_dict(), 'save_model/%s/parts_encoder_l_best_total.pth' % (opt.outf))
                    torch.save(parts_encoder_r.state_dict(), 'save_model/%s/parts_encoder_r_best_total.pth' % (opt.outf))
                    min_loss_total = loss
                #mse_l
                if  loss_mse_l <= min_total_mse_l:
                    print("msel 更新")
                    torch.save(parts_encoder_l.state_dict(), 'save_model/%s/parts_encoder_l_best.pth' % (opt.outf))
                    min_total_mse_l = loss_mse_l
                #mse_r
                if  loss_mse_r <= min_total_mse_r:
                    print("mser 更新")
                    torch.save(parts_encoder_r.state_dict(), 'save_model/%s/parts_encoder_r_best.pth' % (opt.outf))
                    min_total_mse_r = loss_mse_r

        if (epoch+1) % 10 == 0:
            torch.save(parts_encoder_l.state_dict(), 'save_model/%s/parts_encoder_l_epoch_%s.pth' % (opt.outf, epoch))
            torch.save(parts_encoder_r.state_dict(), 'save_model/%s/parts_encoder_r_epoch_%s.pth' % (opt.outf, epoch))
        scheduler.step()
    Writer.close()

    print("----------")
    print("学習終了")
    torch.save(parts_encoder_l.state_dict(), 'save_model/%s/parts_encoder_l_final.pth' % (opt.outf))
    torch.save(parts_encoder_r.state_dict(), 'save_model/%s/parts_encoder_r_final.pth' % (opt.outf))

   
    
   