import torch
import torch.nn as nn
from model.MSA_relative_instance_input import msa_197_768
# from model.cnn_vit_cla2 import cnn_pre
import os
import numpy as np
class conv(nn.Module):#3*3或者1*1卷积，特征图大小不变
    def __init__(self,inchannel,outchannel,n):
        super(conv, self).__init__()
        padding=1 if n==3 else 0
        self.conv=nn.Sequential(
        nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=n, padding=padding),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),)
    def forward(self,x):
        return self.conv(x)

class residual_block(nn.Module):#通道不变的残差块
    def __init__(self,channel=32,n=3):
        super(residual_block, self).__init__()
        self.conv1=conv(inchannel=channel,outchannel=channel,n=n)
        self.conv2= conv(inchannel=channel, outchannel=channel, n=n)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(x+self.conv2(self.conv1(x)))

class one_patch_cnn(nn.Module):#input B*1*16*16
    def __init__(self,num_res = 2):
        super(one_patch_cnn, self).__init__()
        self.num_res = num_res
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            *[residual_block(channel=32) for i in range(self.num_res)],
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[residual_block(channel=64) for i in range(self.num_res)],
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            *[residual_block(channel=32) for i in range(self.num_res)],
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.patch_cnn(x)#B*3*16*16

class patch_cnn(nn.Module):#input B*196*16*16
    def __init__(self):
        super(patch_cnn, self).__init__()
        self.one_patch = one_patch_cnn()
    def forward(self,x):
        B,N,H,W = x.shape
        x = x.view(-1,1,*x.shape[2:]) #(B*N)*1*H*W
        x=self.one_patch(x)#(B*N)*3*H*W
        x=x.view(B,N,-1)
        return x

class MIL_embedding(nn.Module):  # input B*196*768
    def __init__(self):
        super(MIL_embedding, self).__init__()
        self.MIL_Prep1 = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            # torch.nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(128),
            #torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
             nn.Dropout(0.1)
        )
        self.MIL_Prep2 = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            # torch.nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(128),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Sigmoid(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.MIL_Prep1(x) *self.MIL_Prep2(x) # output B*196*128


class MIL_aggregation(nn.Module):  # input B*196*768
    def __init__(self):
        super(MIL_aggregation, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.aggregation = torch.nn.Sequential(  # input B*196*128=>output B*196*1
            nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            torch.nn.ReLU(inplace=True),
        )
        self.mil_emb = MIL_embedding()
        self.sig = nn.Sigmoid()
    def forward(self, x,label):
        x_emb = self.mil_emb(x)  # B*196*128
        x_ag = self.aggregation(x_emb).permute(0, 2, 1)  # B*1*196
        x_ag = nn.functional.softmax(x_ag, dim=2)  # B*1*196

        #pseudo_label = torch.zeros((x.shape[0],196,1),device=torch.device('cuda:0'))

        # pseudo_label = x_ag.permute(0,2,1).clone()

        # print(pseudo_label.shape)
        sigma = torch.zeros((x.shape[0],196),device=torch.device('cuda:0'))
        # print(label.shape)
        label = label.squeeze()
        # print(label)
        if self.training:
            for i in range(x.shape[0]):
                for j in range(196):
                    # print(label[i,j])
                    if label[i,j] > 0.5:
                        theta = 1-label[i,j]
                    else:
                        theta = label[i,j]
                    sigma[i,j] = (1-theta)*(1-theta)
            # print(x_ag.shape)
            # print(sigma.shape)
            # print(sigma)
            x_ag = (sigma*x_ag.squeeze()).unsqueeze(dim=1)
            # print(x_ag.shape)
            out = torch.bmm(x_ag, x_emb)
            return out
        else:
            out = torch.bmm(x_ag, x_emb)
            return out
        # if self.training:
        #     '''
        #     for i in range(x.shape[0]):
        #         # lam = np.random.beta(8, 2)
        #         if label[i] == 1:
        #             pseudo_label[i:i+1]=self.sig((x_ag.permute(0,2,1)[i:i+1]-x_ag[i:i+1].min())/(x_ag[i:i+1].max()-x_ag[i:i+1].min()+1e-8))
        #         else:
        #             # lam = np.random.beta(2,8)
        #             # pseudo_label[i:i+1]=1-lam
        #             pseudo_label[i:i+1] = 1e-8
        #     '''
        #     for i in range(x.shape[0]):  # 遍历每张图像
        #         for j in range(pseudo_label.shape[1]):  # 遍历每个像素点
        #             # lam = np.random.beta(8, 2)
        #             lam = np.random.beta(8, 2)
        #             if label[i] == 1:
        #                 pseudo_label[i, j, 0] = lam
        #             else:
        #                 pseudo_label[i, j, 0] = 1 - lam
        #     return out,pseudo_label.squeeze()#B*1*128,B*196
        # else:return out#B*1*128

class MIL_vit(nn.Module):  #
    def __init__(self):
        super(MIL_vit, self).__init__()
        self.conv_pre = patch_cnn()#[B,196,768]
        #self.conv_pre = cnn_pre()
        self.msa = msa_197_768()
        self.MIL = MIL_aggregation()
        self.bag_classification = nn.Sequential(
            nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.instance_classification_pri = nn.Sequential(
            nn.Linear(768,128),
            torch.nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.instance_classification_aux = nn.Sequential(
            nn.Linear(768,128),
            torch.nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(768,128),
            torch.nn.ReLU(inplace=True),)
    def forward(self, x,label):
        x_cnn = self.conv_pre(x)#[B,196,768]
        x_msa = self.msa(x_cnn)  # B*197*768
        x_cla_fea = x_msa[:, 0, :]  # B*768
        x_mil_fea = x_msa[:, 1:, :]  # B*196*768
        x_cla_fea = self.linear(x_cla_fea)#B*128
        #print(pseudo_label.shape)
        if self.training:
            # bag_fea, pseudo_label = self.MIL(x_mil_fea, label)[0].squeeze(), self.MIL(x_mil_fea, label)[1]  # B*128,B*196
            bag_fea = self.MIL(x_mil_fea, label).squeeze()
            self.x0 = x_mil_fea
            return self.bag_classification(x_cla_fea),self.bag_classification(bag_fea),self.instance_classification_pri(x_mil_fea).squeeze(),self.instance_classification_aux(x_mil_fea).squeeze()#B*2,B*2,B*196,B*196,B*196
        else:
            bag_fea = self.MIL(x_mil_fea, label).squeeze()
            return self.bag_classification((x_cla_fea + bag_fea)/2) #B*2
            #cla = torch.cat([x_cla_fea,x_mil_fea],axis = -1)

    def get_fea(self):
        return self.x0
if __name__ == "__main__":
    a=torch.randn(4,196,16,16)
    #a = torch.randn(4, 3, 224, 224)
    # label = torch.tensor([1,0,1,0],dtype=torch.long)
    label = torch.randn(4,196)
    model =MIL_vit()
    model.train()
    total_num = sum(p.numel() for p in model.parameters())
    b,c,instance_fea1,instance_fea2=model(a,label)
    # x0 = model.get_fea()
    # print(instance_fea1.shape,instance_fea2.shape)
    # print(instance_fea1)
    # print(instance_fea2)
