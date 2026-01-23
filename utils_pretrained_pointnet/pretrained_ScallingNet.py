"""
Parts → Hand(format)
事前学習した手形状生成VAEを用いてEncoderを学習
対称性のある形状(両手付き)の様な物体はパーツセグメンテーションが難しい
→ 教師ラベルを用いて学習
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
from model_pointnet import ScalingNet

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='ScalingNet', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="neuralnet_dataset_unity", help="dataset path")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    opt = parser.parse_args()
    print(opt)
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    os.makedirs('save_model/%s' % (opt.outf), exist_ok=True)

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
        data_augmentation=True)
    
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
    scaleNet = ScalingNet().cuda()
    optimizer = optim.Adam([{'params': scaleNet.parameters()}], lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    num_batch = len(dataset) / opt.batchSize
    min_loss = 100

    for epoch in range(opt.nepoch):
        #scheduler.step()
        for i, data in enumerate(dataloader, 0):
            #print(i, data)
            optimizer.zero_grad()
            scaleNet.train()
            points, target, input_hands, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
            batch_size = points.size(0)
            points, hand_scale = points.cuda(), hand_scale.reshape(batch_size, 2).cuda() 
            points = points.transpose(2, 1)
            pred_scale = scaleNet(points)
            #
            loss = F.mse_loss(hand_scale, pred_scale, reduction="sum") / batch_size
            loss.backward()
            optimizer.step()
            print('[%d: %d/%d] loss: %f' % (epoch, i, num_batch, loss.item()))
            #loss
            Writer.add_scalars("tensorboad/loss",{"train":loss.item()},epoch)

            #eval()
            if (i+1) % 9 == 0:
                j, data = next(enumerate(testdataloader,0))
                scaleNet.eval()
                points, target, input_hands, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data
                batch_size = points.size(0)
                points, hand_scale = points.cuda(), hand_scale.reshape(batch_size, 2).cuda() 
                points = points.transpose(2, 1)
                pred_scale = scaleNet(points)
                loss = F.mse_loss(hand_scale, pred_scale, reduction="sum") / batch_size
                print('[%d: %d/%d] %s loss %f' % (epoch, i, num_batch, blue("test"), loss.item())) 
                #loss
                Writer.add_scalars("tensorboad/loss",{"eval":loss.item()},epoch)
                #model save 
                if loss < min_loss:
                    print("----------")
                    print("min_lossを更新")
                    torch.save(scaleNet.state_dict(), 'save_model/%s/scaleNet_best.pth' % (opt.outf))
                    min_loss = loss
        
        scheduler.step()
    Writer.close()
    print("----------")
    print("学習終了")
    torch.save(scaleNet.state_dict(), 'save_model/%s/scaleNet_final.pth' % (opt.outf))


   
    
   