"""
Training script for Parts2Gesture contrastive learning models.

Trains PointNet-based models for gesture recognition from 3D point cloud parts.
"""

from __future__ import print_function
import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset_format_xy import ShapeNetDataset
from model_pointnet import PointNetDenseCls, feature_transform_regularizer
import shutil

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    
    parser.add_argument('--outf', type=str, default='ScalingNet', help='output folder')
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

    os.makedirs('save_model/%s' % (opt.outf), exist_ok=True)

    if os.path.exists("Log_tensorboard/"+opt.outf):
        shutil.rmtree("Log_tensorboard/"+opt.outf)
    Writer = SummaryWriter(log_dir="Log_tensorboard/"+opt.outf)

    dataset = ShapeNetDataset(
        root=opt.dataset,
        data_augmentation=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        split='val',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes
    print('classes', num_classes)

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model_pointnet = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)
    optimizer = optim.Adam([{"params":model_pointnet.parameters()}], lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    model_pointnet.cuda()

    num_batch = len(dataset) / opt.batchSize

    min_loss = 100
    max_acc = 0
    for epoch in range(opt.nepoch):
        #scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target, hands, filename ,loss_weight = data
            points = points.transpose(2, 1)
            partsseg_target = target
            points, target, loss_weight = points.cuda(), target.cuda() ,loss_weight.cuda()
            optimizer.zero_grad()
            model_pointnet = model_pointnet.train()
            pred, trans, trans_feat,all_feat = model_pointnet(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss =F.nll_loss(pred, target , weight = loss_weight)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print("p",points.shape,"t",target.shape,"pred",pred_choice,"trans",trans.shape,"tansfeat",trans_feat)
            partseg_acc = correct.item()/float(opt.batchSize * 2048)*100
            loss_partseg = loss.item()
            print('[%d: %d/%d] parts segmentation loss: %f' % (epoch, i, num_batch, loss.item()))
            print('[%d: %d/%d] parts segmentation accuracy: %f' % (epoch, i, num_batch, partseg_acc))

            Writer.add_scalars("tensorboad/loss",{"train":loss},epoch)
            Writer.add_scalars("tensorboad/acc_partseg",{"train":partseg_acc},epoch)
            #eval
            if (i+1) % 9 == 0:
                j, data = next(enumerate(testdataloader,0))
                points, target, hands, label, loss_weight = data
                points = points.transpose(2, 1)
                points, target , loss_weight = points.cuda(), target.cuda()
                model_pointnet = model_pointnet.eval()
                pred, trans, trans_feat,all_feat = model_pointnet(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                loss =F.nll_loss(pred, target , weight = loss_weight)
                if opt.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001

                pred_choice = pred.data.max(1)[1]
                pred_np = pred_choice.cpu().data.numpy()
                points=points.transpose(1, 2).cpu().data.numpy()
                pred_np=pred_np.reshape(opt.batchSize,2048,1)      
                #parts segmentation acc
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print("p",points.shape,"t",target.shape,"pred",pred_choice,"trans",trans.shape,"tansfeat",trans_feat)
                val_acc = correct.item()/float(opt.batchSize * 2048)*100
                loss_partseg = loss.item()

                print('[%d: %d/%d] %s total-test loss: %f' % (epoch, i, num_batch, blue("test"), loss.item()))
                print('[%d: %d/%d] %s partssesg loss: %f accuracy: %f' % (epoch, i, num_batch, blue("test"),loss_partseg, val_acc)) 
                Writer.add_scalars("tensorboad/loss",{"eval":loss},epoch)
                Writer.add_scalars("tensorboad/acc_partseg",{"eval":val_acc},epoch)

                #model save
                if loss < min_loss:
                    print("----------")
                    print("min_lossを更新")
                    torch.save(model_pointnet.state_dict(), 'save_model/%s/pointnet_acc_partseg_best.pth' % (opt.outf))
                    min_loss = loss
                if val_acc > max_acc:
                    print("----------")
                    print("max_accを更新")
                    torch.save(model_pointnet.state_dict(), 'save_model/%s/pointnet_loss_partseg_best.pth' % (opt.outf))
                    max_acc = val_acc

            scheduler.step()
    Writer.close()

    print("----------")
    print("学習終了")
    torch.save(model_pointnet.state_dict(), 'save_model/%s/pointnet_final.pth' % (opt.outf))
