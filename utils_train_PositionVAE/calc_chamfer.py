from __future__ import print_function
import argparse
import os
import random
import torch
import matplotlib.pyplot as plt
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset_format_xy import ShapeNetDataset_format
from model4 import PointNetDenseCls, feature_transform_regularizer , PartsNet2_vae, PartsNet_2
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from model_format import VAE, PartsNet_vae_ges, QuaterionVAE, WristGenerater
from torch.autograd import Variable
from utils_Pretrained_Hand.visualization import *
from caclulate_method import *

pointnet_classifier = PointNetDenseCls(k=3, feature_transform=None)
state_dict_pointnet = torch.load("save_pretrained_partsseg/pointnet_model_sotuken_acc_partseg_best.pth", weights_only=True)
pointnet_classifier.load_state_dict(state_dict_pointnet)
pointnet_classifier.eval()
#train model 
"rotation matrix NN"

sita_generater_l, sita_generater_r = QuaterionVAE(), QuaterionVAE()
state_sita_l = torch.load("train_g2p/sita_generater_l_sotuken_epoch.pth", weights_only=True)
state_sita_r = torch.load("train_g2p/sita_generater_r_sotuken_epoch.pth", weights_only=True)
sita_generater_l.load_state_dict(state_sita_l)
sita_generater_r.load_state_dict(state_sita_r)
sita_generater_l.eval(), sita_generater_r.eval()


wrist_generater = WristGenerater()
state_wrist = torch.load("train_g2p/wrist_generater_sotuken_epoch.pth", weights_only=True)
wrist_generater.load_state_dict(state_wrist)
wrist_generater.eval()
#学習済みvae
"initial hand vae"

parts_encoder_classifier_l, parts_encoder_classifier_r = PartsNet_vae_ges(), PartsNet_vae_ges()
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



parser = argparse.ArgumentParser()

choice="Mug"

parser.add_argument('--model', type=str, default='C:/Users/yokada/Desktop/new/pointnet.pytorch/segqq/seg_model_'+choice+'_loss.pth', help='model path')
parser.add_argument('--idx', type=int, default= 230, help='model index')
parser.add_argument('--dataset', type=str, default='neuralnet_dataset_unity', help='dataset path')
parser.add_argument('--class_choice', type=str, default=choice, help='class choice')
opt = parser.parse_args()
print(opt)


d = ShapeNetDataset_format(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split="search",
    data_augmentation = True)

point_set, seglabel, hand_target, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = d[opt.idx]

"hand_target : GT"
def evaluuate(data):
    point_set, seglabel, hand_target, label, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist = data


    point = point_set.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat= pointnet_classifier(point)
    pred_choice = pred.data.max(2)[1].cpu()

    pl=np.array([])
    pr=np.array([])
    parts_l_list=np.array([])
    parts_r_list=np.array([])
    #pred_choice = pred_choice.cpu().data.numpy()
    #label 1=右手　2 = 左手
    print("推測  label 1 ,2 ,0:",np.count_nonzero(pred_choice==1),np.count_nonzero(pred_choice==2),np.count_nonzero(pred_choice==0))
    print("答え  label 1 ,2 ,0:",np.count_nonzero(seglabel==1),np.count_nonzero(seglabel==2),np.count_nonzero(seglabel==0))

    print(pred_choice.shape)
    pred_choice=pred_choice[0]
    print(pred_choice.shape)

    if np.count_nonzero(pred_choice==2)<=10:   
        print("左手パーツ未検出") 
        target_l=seglabel
    else:
        target_l=pred_choice
    if np.count_nonzero(pred_choice==1)<=10:
        print("右手パーツ未検出")
        target_r=pred_choice

    for j in range(2048):
        
        if target_l[j]==2:
            parts_l_list=np.append(parts_l_list,point_set[j])
        if target_l[j]==1:
            parts_r_list=np.append(parts_r_list,point_set[j])

    while len(parts_l_list)<=(3 * 256):
        #print("augment")
        add_list=parts_l_list*1.01
        parts_l_list=np.append(parts_l_list,add_list)
    while len(parts_r_list)<=(3 * 256):
        #print("augment")
        add_list=parts_r_list*1.01
        parts_r_list=np.append(parts_r_list,add_list)

    parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
    parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)   

    pl = farthest_point_sampling(parts_l_list, target_num=256)   
    pr = farthest_point_sampling(parts_r_list, target_num=256)  

    print("chamfer distance")
    print(calc_mean_chamferD(pl.reshape(256, 3)))
    print(calc_mean_chamferD(pr.reshape(256, 3)))


    pl=pl.reshape(1,256,3).astype(np.float32)
    pr=pr.reshape(1,256,3).astype(np.float32)
    #指を0,0,0にするから移動させよう。(後に使う)
    pl_move = np.expand_dims(np.mean(pl, axis = 1), 0)
    pr_move = np.expand_dims(np.mean(pr, axis = 1), 0)

    pl = pl - pl_move
    pr = pr - pr_move

    pl=torch.from_numpy(pl)
    pr=torch.from_numpy(pr)
    pl=pl.transpose(2,1)
    pr=pr.transpose(2,1)

    print(pl.shape)

    #################chamfer distance


    parts_l, parts_r, all_feat =  pl, pr, all_feat

    pred_wrist = wrist_generater(all_feat)
    
    wrist_l, wrist_r = np.split(pred_wrist, 2, axis=1)

    
    D_l, z_l = calc_Dandz(wrist_l)
    D_r, z_r = calc_Dandz(wrist_r)

    wrist_l_t, wrist_r_t = np.split(wrist, 2, axis=0)
    wrist_l_t, wrist_r_t =  wrist_l_t.view(-1, 3), wrist_r_t.view(-1, 3)

    Dt_l, zt_l = calc_Dandz(wrist_l_t)
    Dt_r, zt_r = calc_Dandz(wrist_r_t)

    loss_wrist_l = F.mse_loss(D_l, Dt_l) + F.mse_loss(z_l, zt_l) 
    loss_wrist_r = F.mse_loss(D_r, Dt_r) + F.mse_loss(z_r, zt_r) # mean

    print("wrist mseloss:", loss_wrist_l, loss_wrist_r)
    
    print("wrist")
    print("pred, target", wrist_l, wrist_l_t)
    print("pred, target", wrist_r, wrist_r_t)

    hscale_l, hscale_r = np.split(hand_scale, 2, axis=0)

    #手形状生成
    pf_l, mu_l, logvar_l = parts_encoder_classifier_l(pl, all_feat)
    pf_r, mu_r, logvar_r = parts_encoder_classifier_r(pr, all_feat)  
    pred_handl = vae_classifier_l.finetune(pf_l)
    pred_handr = vae_classifier_r.finetune(pf_r)

    print("AAA")

    #回転行列生成
    R_l, R_mu_l, R_logvar_l = sita_generater_l(pl, all_feat)
    R_r, R_mu_r, R_logvar_r = sita_generater_r(pr, all_feat)

    R_l, R_r = z_rotation_matrix(R_l), z_rotation_matrix(R_r)
    wrist_format = torch.tensor([0.5, 0.5, 0.5])
    

    pred_handl, pred_handr = pred_handl.view(1, -1, 3) - wrist_format , pred_handr.view(1, -1, 3) - wrist_format

    wrist_l, wrist_r = wrist_l.view(1, -1, 3), wrist_r.view(1, -1, 3)
    #print(pred_handl.shape, R_l.shape, wrist_l.shape)
    pred_ges_l_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handl], dim=1)
    pred_ges_r_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handr], dim=1)



    pred_ges_l = pred_ges_l_format @ R_l.transpose(1,2) + wrist_l 
    pred_ges_r = pred_ges_r_format @ R_r.transpose(1,2) + wrist_r
    
    pred_ges_l_format = pred_ges_l_format @ R_l.transpose(1,2) + wrist_l 


    #print(pred_ges_l)


    hscale_l, hscale_r = hscale_l, hscale_r
    pred_ges_l = pred_ges_l / hscale_l
    pred_ges_r = pred_ges_r / hscale_r

    target_hand_l, target_hand_r = np.split(hand_target, 2, axis=0)
    plc, prc = torch.from_numpy(pl_move.astype(np.float32)), torch.from_numpy(pr_move.astype(np.float32))

    loss_grab_l = Loss_Grable(pred_ges_l, target_hand_l, plc, D_threshold=0.1).mean() 
    loss_grab_r = Loss_Grable(pred_ges_r, target_hand_r, prc, D_threshold=0.1).mean() 

    print("Grable Loss:", loss_grab_l, loss_grab_r)
    return point_set, pred_choice, hand_target, pred_ges_l, pred_ges_r, pred_wrist, parts_l_list, parts_r_list, pred_ges_l_format, pred_ges_r_format



fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax.set_title('ans', fontsize=20) 
plt.axis('off')
plt.tight_layout()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax1.set_title('partseg', fontsize=20) 
plt.axis('off')
plt.tight_layout()
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(-1,1)

fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax2.set_title('parts', fontsize=20)
plt.axis('off')
plt.tight_layout()
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(-1,1)

fig3, ax3 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax3.set_title('output_ges', fontsize=20) 
plt.axis('off')
plt.tight_layout()
ax3.set_xlim(-1.5,1.5)
ax3.set_ylim(-1.5,1.5)
ax3.set_zlim(-1,1)

fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax4.set_title('parts_l', fontsize=20) 
plt.axis('off')
plt.tight_layout()
ax4.set_xlim(-1,1)
ax4.set_ylim(-1,1)
ax4.set_zlim(-1,1)

fig5, ax5 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax5.set_title('parts_r ', fontsize=20)
plt.axis('off')
plt.tight_layout()
ax5.set_xlim(-1,1)
ax5.set_ylim(-1,1)
ax5.set_zlim(-1,1)

fig6, ax6 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax6.set_title('pts-ges', fontsize=20) 
plt.axis('off')
plt.tight_layout()
ax6.set_xlim(-1.5,1.5)
ax6.set_ylim(-1.5,1.5)
ax6.set_zlim(-1,1)

fig7, ax7 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
#ax7.set_title('input', fontsize=20) 
plt.axis('off')
plt.tight_layout()
ax7.set_xlim(-1,1)
ax7.set_ylim(-1,1)
ax7.set_zlim(-1,1)


fig8, ax8 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax8.set_title('predict gesture', fontsize=20) 
plt.axis('off')
plt.tight_layout()




point_set, pred_choice, hand_target, pred_ges_l, pred_ges_r, pred_wrist, pl, pr, pred_ges_l_format, pred_ges_r_format  = evaluuate(d[opt.idx])

point_set=point_set.numpy()
drawparts(point_set, ax=ax, parts="")
drawpts(point_set,pred_choice,ax=ax1)
drawparts(pl, ax=ax2, parts="left" )
drawparts(pr, ax=ax2, parts="right")

handl_gt, handr_gt = np.split(hand_target, 2, axis=0)
handr_gt, handr_gt = handl_gt.numpy(), handr_gt.numpy()
drawhand(handl_gt, color= "red", ax = ax3, handinf=handinf)
drawhand(handr_gt, color= "blue", ax = ax3, handinf=handinf)

pwrist_l, pwrist_r = np.split(pred_wrist, 2, axis=1)
pwrist_l, pwrist_r = pwrist_l.detach().numpy(), pwrist_r.detach().numpy()
predhand_l, predhand_r = pred_ges_l.view(-1, 3).detach().numpy(), pred_ges_r.view(-1, 3).detach().numpy()


drawhand(predhand_l, color= "orange", ax = ax4, handinf=handinf)
drawhand(predhand_r, color= "cyan", ax = ax4, handinf=handinf)

drawparts(point_set, ax=ax5, parts="")
drawhand(handl_gt, color= "red", ax = ax5, handinf=handinf)
drawhand(handr_gt, color= "blue", ax = ax5, handinf=handinf)

drawparts(point_set, ax=ax6, parts="")
drawhand(predhand_l, color= "orange", ax = ax6, handinf=handinf)
drawhand(predhand_r, color= "purple", ax = ax6, handinf=handinf)

drawhand(pred_ges_l_format.view(23, 3).detach().numpy(), color="orange", ax =ax7, handinf=handinf)
plt.show()