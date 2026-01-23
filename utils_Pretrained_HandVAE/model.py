"""
手形状生成は、回転普遍性を持つ→どんなパーツ形状の向きでも「同じ向きの手形状生成するように」->T-Netあり、VAE構造
回転角θは、回転普遍性がいらない→T-netいる→VAE構造でなくてもよい
T-netありは、PointNetfeat_newで管理
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys

class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder,self).__init__()
        self.d1=nn.Linear(66,66)
        self.d2=nn.Linear(66,66)
        self.d3=nn.Linear(66,66)
        self.d4=nn.Linear(66,64)
        self.d5=nn.Linear(64,32)

        self.bn1 = nn.BatchNorm1d(66)
        self.bn2 = nn.BatchNorm1d(66)
        self.bn3 = nn.BatchNorm1d(66)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
    
    def forward(self,x):
        fc1=F.relu(self.bn1(self.d1(x)))
        fc2=F.relu(self.bn2(self.d2(fc1)))
        fc3=F.relu(self.bn3(self.d3(fc2)))
        fc4=F.relu(self.bn4(self.d4(fc3)))
        fc5=F.relu(self.bn5(self.d5(fc4)))
        return fc5

class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder,self).__init__()
        self.up1 = nn.Linear(16,32)
        self.up2 = nn.Linear(32, 64)
        self.up3 = nn.Linear(64, 66)
        self.up4 = nn.Linear(66, 66)
        self.up5 = nn.Linear(66, 66)
        self.up6 = nn.Linear(66, 66)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(66)
        self.bn4 = nn.BatchNorm1d(66)
        self.bn5 = nn.BatchNorm1d(66)

    def forward(self, x): 
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        x = F.relu(self.bn3(self.up3(x)))
        x = F.relu(self.bn4(self.up4(x)))
        x = F.relu(self.bn5(self.up5(x)))
        x = F.sigmoid(self.up6(x))
        return x

class HandVAE(nn.Module):
    def __init__(self):
        super(HandVAE, self).__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.fc_mu = nn.Linear(32,16)
        self.fc_logvar = nn.Linear(32,16)

    def encode(self,x):
        fc5 = self.encoder(x)
        mu = self.fc_mu(fc5)
        logvar = self.fc_logvar(fc5)
        #logvar = F.softplus(logvar)
        return mu, logvar
    
    def reparameterize(self, mu ,logvar):
        epsilon = torch.rand_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + epsilon * std
        return z
    
    def forward(self,x):
        mu, logvar= self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z)
        return pred, mu, logvar
    
    def loss (self, x, target = None, beta = None):
        #beta = 0.1 best?
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z) 
        KL_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        if target == None:
            reconstraction_loss = F.mse_loss(x, pred, reduction = "sum") / x.size(0)
        else:
            reconstraction_loss = F.mse_loss(pred, target ,reduction = "sum") / x.size(0)
       
        if beta == None:
            loss = reconstraction_loss + KL_loss
        else:
            loss = beta * reconstraction_loss + KL_loss
        return loss, reconstraction_loss, KL_loss
    
    def get_hidden_z(self,x):
        mu, logvar= self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def test(self, samples ):
        #samples = int
        random_z = torch.randn(samples,16)
        #pred = self.decoder(z)
        random_create = self.decoder(random_z)
        return random_create, random_z
    
    def test2(self, random_z ):
        #samples = int
        random_create = self.decoder(random_z)
        return random_create, random_z
    
    def finetune(self, x):
        #input features x : 16d
        pred = self.decoder(x)
        return pred

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        #print(x)
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        #print(x.shape)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class Handscale_NN(torch.nn.Module):
    def __init__(self,feature_transform=False):
        super(Handscale_NN, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64,6)
        self.dropout = nn.Dropout(p=0.3)

        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(6)

    def forward(self, allfeat):
        #all_feat:1024
        x = F.relu(self.bn2(self.fc2(allfeat)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        return x 
    
class PointNetfeat_new(nn.Module):
    def __init__(self):
        super(PointNetfeat_new, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x, isgesture = True):
        #print(x.shape)
        if isgesture == True:
            x = self.stn(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class WristGenerater(torch.nn.Module):
    #input: allfeat, out, wrist
    #Loss: zが不変、D(x,y)^2が不変
    def __init__(self,feature_transform=False):
        super(WristGenerater, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 6)

        self.dropout = nn.Dropout(p=0.3)

        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(6)

    def forward(self, allfeat):
        #all_feat:1024
        x = F.relu(self.bn2(self.fc2(allfeat)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.bn6(self.fc6(x))

        return x
        
class WristGenerater2(torch.nn.Module):
    #input: allfeat, out, wrist
    #Loss: zが不変、D(x,y)^2が不変
    def __init__(self,feature_transform=False):
        super(WristGenerater2, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64,32)
        self.fc7 = nn.Linear(16, 6)

        self.fc_mu = nn.Linear(32,16)
        self.fc_logvar = nn.Linear(32,16)

        self.dropout = nn.Dropout(p=0.3)

        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(6)

    
    def reparameterize(self, mu ,logvar):
        epsilon = torch.rand_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + epsilon * std
        return z

    def forward(self, allfeat):
        #all_feat:1024
        x = F.relu(self.bn2(self.fc2(allfeat)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        x = self.reparameterize(mu,logvar)

        x = self.bn7(self.fc7(x))

        return x, mu, logvar
    
class PartsEncoder_w_TNet(torch.nn.Module):
    def __init__(self,feature_transform=False):
        super(PartsEncoder_w_TNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64,32)
        self.dropout = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)

        self.fc_mu = nn.Linear(32,16)
        self.fc_logvar = nn.Linear(32,16)

    def reparameterize(self, mu ,logvar):
        epsilon = torch.rand_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + epsilon * std
        return z
    
    def forward(self, parts, all_feat):
        x = self.feat(parts, True)
        #x:1024,all_feat:1024
        x = torch.cat([x,all_feat],dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        x = self.reparameterize(mu,logvar)
        return x, mu, logvar

class PartsEncoder_wo_TNet(torch.nn.Module):
    def __init__(self,feature_transform=False):
        super(PartsEncoder_wo_TNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64,32)
        self.dropout = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)

    def forward(self, parts, all_feat):
        x = self.feat(parts, False) #T-Netを使わない→回転や移動普遍性を考慮しない
        x = torch.cat([x,all_feat],dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        #x:1024,all_feat:1024
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        return x

def quaternion_to_rotation_matrix(q):
    """
    q: tensor of shape (B, 4)  最後の次元が [w, x, y, z]
       単位クォータニオンであることを前提とします
    return: tensor of shape (B, 3, 3)
    """
    # 要素ごとに分解
    w, x, y, z = q.unbind(-1)

    # バッチごとに 3x3 回転行列を計算
    R = torch.zeros(q.size(0), 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)

    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)

    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)

    return R

class Quaterion_Decoder(torch.nn.Module):
    def __init__(self):
        super(Quaterion_Decoder).__init__()
        self.fn1 = nn.Linear(16, 8)
        self.fn2 = nn.Linear(8, 4)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fn1(x)))
        x = F.tanh(self.bn2(self.fn2(x)))
        x = F.normalize(x, dim=-1)

        # 変換
        R = quaternion_to_rotation_matrix(x)
        return R

#############層を増やす###############

class Quaterion_Decodernew(nn.Module):
    def __init__(self):
        super(Quaterion_Decodernew, self).__init__()
        self.fn1 = nn.Linear(16,32)
        self.fn2 = nn.Linear(32,32)
        self.fn3 = nn.Linear(32,32)
        self.fn4 = nn.Linear(32, 2)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fn1(x)))
        x = F.tanh(self.bn2(self.fn2(x)))
        x = F.tanh(self.bn3(self.fn3(x)))
        x = F.tanh(self.bn4(self.fn4(x)))
        x = F.normalize(x, dim=-1)
        return x

class WristPosition_Decoder(nn.Module):
    def __init__(self):
        super(WristPosition_Decoder, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4= nn.Linear(32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.bn3(self.fc3(x))
        x = 2 * F.tanh(self.fc4(x))
        return x

class QuaterionVAE5new(nn.Module):
    def __init__(self):
        super(QuaterionVAE5new, self).__init__()
        self.encoder = PartsEncoder_wo_TNet()
        self.decoder = Quaterion_Decodernew() #quaterion
        self.fc_mu1 = nn.Linear(32,16)
        self.fc_logvar1 = nn.Linear(32,16)
        self.decode_wrist = WristPosition_Decoder() # wrist position
        
    def encode(self,x,allfeat):
        x = self.encoder(x, allfeat)
        mu_r = self.fc_mu1(x)
        logvar_r = self.fc_logvar1(x)
        #logvar = F.softplus(logvar)
        return mu_r, logvar_r
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def reparameterize_multi(self, mu, logvar, N):
        std = torch.exp(0.5 * logvar)          # [B, D]
        eps = torch.randn(mu.size(0), N, mu.size(1),
                        device=mu.device)   # [B, N, D]
        return mu.unsqueeze(1) + eps * std.unsqueeze(1)

    
    def forward(self, x, allfeat, N):
        mu, logvar = self.encode(x, allfeat)
        z = self.reparameterize_multi(mu, logvar, N)  # [B, N, D]
        B, N, D = z.shape
        z_flat = z.view(B * N, D)
        pred = self.decoder(z_flat).view(B, N, -1)
        pred_wrist = self.decode_wrist(z_flat).view(B, N, -1)
        return pred, pred_wrist, self.kld_loss(mu, logvar), z.detach().cpu().numpy()

    def kld_loss(self, mu, logvar):
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return kld
    
    def get_z(self,x, allfeat):
        mu_r, logvar_r = self.encode(x, allfeat)
        z_r = self.reparameterize(mu_r, logvar_r)
        pred_r = self.decoder(z_r)
        pred_wrist = self.decode_wrist(z_r)
        return z_r.detach().cpu().numpy(), pred_r, pred_wrist 
    
    def get_muvar(self,x, allfeat):
        mu_r, logvar_r = self.encode(x, allfeat)
        return mu_r, logvar_r
    
    def get_mu_sigma(self, x, allfeat):
        return self.encode(x, allfeat)


