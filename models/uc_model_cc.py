import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MSA_relative_instance_input_qsp import msa_197_768
from torch.distributions import Dirichlet

class conv(nn.Module):
    def __init__(self,inchannel,outchannel,n):
        super(conv, self).__init__()
        padding=1 if n==3 else 0
        self.conv=nn.Sequential(
        nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=n, padding=padding),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),)
    def forward(self,x):
        return self.conv(x)

class residual_block(nn.Module):
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
        return self.patch_cnn(x)   # B*3*16*16

class patch_cnn(nn.Module):# input B*196*16*16
    def __init__(self):
        super(patch_cnn, self).__init__()
        self.one_patch = one_patch_cnn()
    def forward(self,x):
        B,N,H,W = x.shape
        x = x.view(-1,1,*x.shape[2:]) # (B*N)*1*H*W
        x=self.one_patch(x) #(B*N)*3*H*W
        x=x.view(B,N,-1)
        return x


class EvidenceClassifier(nn.Module):        # input B*196*768
    def __init__(self):
        super(EvidenceClassifier, self).__init__()
        self.instance_classification = nn.Sequential(
            nn.Linear(768, 128),
            torch.nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return F.softplus(self.instance_classification(x))



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
        return self.MIL_Prep1(x) *self.MIL_Prep2(x) 


class CausalityScoreModule(nn.Module):
    def __init__(self, input_dim, k):
        super(CausalityScoreModule, self).__init__()
        self.k = k
        self.gcn_layer = GraphConvolution(input_dim, 1) 

    def forward(self, X_i):
        B, N, _ = X_i.shape
        A = torch.stack([self.build_adjacency_matrix(x_i) for x_i in X_i])
        S_i = self.gcn_layer(X_i, A).squeeze(-1)

        F_i_c = [] 
        F_i_nc = [] 
        topk_indices = []

        for i in range(B):
            scores = S_i[i]  
            x = X_i[i]  # [N, D]

            topk_idx = torch.topk(scores, self.k, dim=0)[1]  #[k]
            mask = torch.ones(N, dtype=torch.bool)
            mask[topk_idx] = False
            non_topk_idx = torch.arange(N)[mask]  # [N-k]

            F_i_c.append(x[topk_idx])  # [k, D]
            F_i_nc.append(x[non_topk_idx])  # [N-k, D]
            topk_indices.append(topk_idx)

        topk_indices = torch.stack(F_i_c, dim=0)
        F_i_c = torch.stack(F_i_c, dim=0)
        F_i_nc = torch.stack(F_i_nc, dim=0)

        return S_i, topk_indices, F_i_c.mean(dim=1, keepdim=True), F_i_nc.mean(dim=1, keepdim=True)

    def build_adjacency_matrix(self, X):
        X_centered = X - X.mean(dim=0)
        cov = X_centered @ X_centered.T 
        var = torch.sqrt(torch.sum(X_centered ** 2, dim=1, keepdim=True)) + 1e-6
        corr = cov / (var @ var.T)
        A = torch.abs(corr)
        D = torch.diag_embed(torch.sum(A, dim=-1) ** (-0.5))
        A_norm = D @ A @ D
        return A_norm


class GraphConvolution(nn.Module):
    """Simple GCN layer: H = A_norm X W"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, A_norm):
        return A_norm @ self.linear(X)



class UC_MIL(nn.Module):  #
    def __init__(self):
        super(UC_MIL, self).__init__()
        self.conv_pre = patch_cnn()#[B,196,768]
        #self.conv_pre = cnn_pre()
        self.msa = msa_197_768()
        self.linear = nn.Sequential(
            nn.Linear(768,128),
            torch.nn.ReLU(inplace=True),)
        self.evidence_classifier = EvidenceClassifier()
        self.emb_layer = MIL_embedding()
        self.causality_score_module = CausalityScoreModule(768, 50)
        self.fusion_layer = torch.nn.Linear(256, 128)
        self.bag_classification = nn.Sequential(
            nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )


    def forward(self, x):
        # print(label.shape)
        x_cnn = self.conv_pre(x)#[B,196,768]
        x_msa = self.msa(x_cnn)  # B*197*768
        x_cla_fea = x_msa[:, 0, :]  # B*768
        x_mil_fea = x_msa[:, 1:, :]  # B*196*768
        x_cla_fea = self.linear(x_cla_fea)#B*128

        # USL
        evidence = self.evidence_classifier(x_mil_fea)
        dirichlet_params = self.calculate_dirichlet_params(evidence)
        instance_pred, uncertainty = self.calculate_belief_and_uncertainty(dirichlet_params)

        # CPE
        CauScore, topk_indices, F_c, F_nc = self.causality_score_module(x_mil_fea)
        if x.shape[0] == 1:
            F_c = self.emb_layer(F_c).squeeze().unsqueeze(0)
            F_nc = self.emb_layer(F_nc).squeeze().unsqueeze(0)
        else:
            F_c = self.emb_layer(F_c).squeeze()
            F_nc = self.emb_layer(F_nc).squeeze()
        bag_pred = self.bag_classification((x_cla_fea + F_c)/2)
        # cls_token = self.bag_classification(x_cla_fea)

        return instance_pred, uncertainty, CauScore, topk_indices, F_c, F_nc, bag_pred

    def get_fea(self):
        return self.x0

    def calculate_dirichlet_params(self, evidence):
        I = torch.ones_like(evidence)
        dirichlet_params = evidence + I
        return dirichlet_params

    def calculate_belief_and_uncertainty(self, dirichlet_params):
        pred = dirichlet_params / dirichlet_params.sum(dim=-1, keepdim=True)
        uncertainty = 2 / dirichlet_params.sum(dim=-1, keepdim=True)
        return pred, uncertainty


if __name__ == "__main__":
    a=torch.randn(4,196,16,16)
    # a = torch.randn(4, 3, 224, 224).cuda()
    # label = torch.tensor([1,0,1,0],dtype=torch.long)
    label = torch.randn(4,196)
    model = UC_MIL()
    model.train()
    total_num = sum(p.numel() for p in model.parameters())
    instance_pred, uncertainty, CauScore, topk_indices, F_c, F_nc, bag_pred = model(a,label)
    print(bag_pred)
