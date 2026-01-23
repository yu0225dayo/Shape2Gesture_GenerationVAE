

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

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter

import shutil
from model import HandVAE

handinf=[0,1,2,3,4,18,
         0,5,6,7,19,
         0,8,9,10,20,
         0,11,12,13,21,
         0,14,15,16,17,22]

bone_pairs = [
        (0,1), (1,2), (2,3), (3,4), (4,18),
        (0,5), (5,6), (6,7), (7,19),
        (0,8), (8,9), (9,10), (10,20),
        (0,11), (11,12), (12,13), (13,21),
        (0,14), (14,15), (15,16), (16,17), (17,22)
            ]

#骨格長でLossをとる
def bone_length_loss(pred_output, gt_output, bone_pairs):
    """
    pred_output: Tensor, shape (batch, 69)
    gt_output: Tensor, shape (batch, 69)
    """
    # Reshape to (batch, 23, 3)
    pred_joints = pred_output.view(-1, 23, 3)
    gt_joints = gt_output.view(-1, 23, 3)
    loss = 0
    for i, j in bone_pairs:
        pred_len = torch.norm(pred_joints[:, i] - pred_joints[:, j], dim=1)
        gt_len = torch.norm(gt_joints[:, i] - gt_joints[:, j], dim=1)
        loss += F.l1_loss(pred_len , gt_len,reduce="sum")
    return loss / len(bone_pairs)

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument(
        '--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='pretrained_HandVAE_formatxy', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="hand_vae_data2", help="dataset path")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--beta', type= float , default=0.1,  help="use for beta vae")
    opt = parser.parse_args()
    print(opt)
    if os.path.exists("Log_tensorboard/"+opt.outf):
        shutil.rmtree("Log_tensorboard/"+opt.outf)

    Writer = SummaryWriter(log_dir="Log_tensorboard/"+opt.outf)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    blue = lambda x: '\033[94m' + x + '\033[0m'

    dataset = HandDataset_format(
        root=opt.dataset,
        data_augmentation=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    test_dataset = HandDataset_format(
        root=opt.dataset,
        split='val',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    if not os.path.exists("save_model/"+opt.outf):
        print("AAA")
        os.mkdir("save_model/"+opt.outf)

    handvae_l = HandVAE()
    handvae_r = HandVAE()
    optimizer = optim.Adam([{"params":handvae_l.parameters()},{"params":handvae_r.parameters()}] ,
                                lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda epoch:0.95** epoch)
    handvae_l.cuda()
    handvae_r.cuda()

    num_batch = len(dataset) / opt.batchSize
    min_loss_total_l = 100
    min_loss_total_r = 100
    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()          
            handvae_l = handvae_l.train()
            handvae_r = handvae_r.train()

            #print(i, data)
            hands, nomalhand, hand_format = data
            # 教師は、手首座標(0.5, 0.5, 0.5)親指ベクトルが(1,0,0)に平行な手形状
            target_hand_l, target_hand_r = np.split(hand_format + torch.tensor([0.5, 0.5, 0.5]), 2, axis = 1) 
            target_hand_l, target_hand_r = target_hand_l.reshape(opt.batchSize, 69), target_hand_r.reshape(opt.batchSize, 69) 
            target_hand_l, target_hand_r = target_hand_l[:, 3:], target_hand_r[:, 3:]
            target_hand_l, target_hand_r = target_hand_l.cuda(), target_hand_r.cuda()
            hand_l_input, hand_r_input = target_hand_l, target_hand_r

            pred_l , _, _ = handvae_l(hand_l_input)
            pred_r , _, _ = handvae_r(hand_r_input)

            loss_total_l , mse_l, kl_l = handvae_l.loss(hand_l_input, beta = opt.beta)
            loss_total_r , mse_r, kl_r = handvae_r.loss(hand_r_input, beta = opt.beta)

            #loss 
            loss_total = ( loss_total_l + loss_total_r ) /2 
            loss_mse = ( mse_l + mse_r ) /2 
            loss_kl = ( kl_l + kl_r ) /2
            loss_total.backward()
            optimizer.step()
            
            print('[%d: %d/%d] loss_mse_l: %f , loss_mse_r: %f' % (epoch, i, num_batch, mse_l.item(),mse_r.item()))
                            
            Writer.add_scalars("tensorboad/loss_total",{"train":loss_total.item()},epoch)
            Writer.add_scalars("tensorboad/loss_mse_l",{"train":mse_l.item()},epoch)
            Writer.add_scalars("tensorboad/loss_kl_l",{"train":kl_l.item()},epoch)
            Writer.add_scalars("tensorboad/loss_mse_r",{"train":mse_r.item()},epoch)
            Writer.add_scalars("tensorboad/loss_kl_r",{"train":kl_r.item()},epoch)
            #Writer.add_scalars("tensorboad/loss_kl",{"train":loss_bone_length.item()},epoch)

            #eval()
            if (i+1) % 9 == 0:
                j, data = next(enumerate(testdataloader,0))
                handvae_l = handvae_l.eval()
                handvae_r = handvae_r.eval()

                hands, nomalhand, hand_format = data
                # 教師は、手首座標(0.5, 0.5, 0.5)親指ベクトルが(1,0,0)に平行な手形状
                target_hand_l, target_hand_r = np.split(hand_format + torch.tensor([0.5, 0.5, 0.5]), 2, axis = 1) 
                target_hand_l, target_hand_r = target_hand_l.reshape(opt.batchSize, 69), target_hand_r.reshape(opt.batchSize, 69) 
                target_hand_l, target_hand_r = target_hand_l[:, 3:], target_hand_r[:, 3:]
                target_hand_l, target_hand_r = target_hand_l.cuda(), target_hand_r.cuda()
                hand_l_input, hand_r_input = target_hand_l, target_hand_r

                pred_l , _, _ = handvae_l(hand_l_input)
                pred_r , _, _ = handvae_r(hand_r_input)

                loss_total_l , mse_l, kl_l = handvae_l.loss(hand_l_input, beta = opt.beta)
                loss_total_r , mse_r, kl_r = handvae_r.loss(hand_r_input, beta = opt.beta)

                #loss 
                loss_total = ( loss_total_l + loss_total_r ) /2 
                loss_mse = ( mse_l + mse_r ) /2 
                loss_kl = ( kl_l + kl_r ) /2

                print('[%d: %d/%d] %s loss_mse_l: %f ,loss_mse_r %f' % (epoch, i, num_batch, blue("test"), mse_l.item(), mse_r.item()))
                Writer.add_scalars("tensorboad/loss_total",{"val":loss_total.item()},epoch)
                Writer.add_scalars("tensorboad/loss_mse_l",{"val":mse_l.item()},epoch)
                Writer.add_scalars("tensorboad/loss_kl_l",{"val":kl_l.item()},epoch)
                Writer.add_scalars("tensorboad/loss_mse_r",{"val":mse_r.item()},epoch)
                Writer.add_scalars("tensorboad/loss_kl_r",{"val":kl_r.item()},epoch)

                if loss_total_l <= min_loss_total_l:
                    print("Loss L 更新")
                    torch.save(handvae_l.state_dict(), 'save_model/%s/vae_l_best.pth' % (opt.outf))
                    min_loss_total_l = loss_total_l

                if loss_total_r <= min_loss_total_r:
                    print("Loss R 更新")
                    torch.save(handvae_r.state_dict(), 'save_model/%s/vae_r_best.pth' % (opt.outf))
                    min_loss_total_r = loss_total_r

        scheduler.step()
    Writer.close()

    print("----------")
    print("学習終了")
    torch.save(handvae_l.state_dict(), 'save_model/%s/vae_l_final.pth' % (opt.outf))
    torch.save(handvae_r.state_dict(), 'save_model/%s/vae_r_final.pth' % (opt.outf))
    









    




    

    

        
        
        



