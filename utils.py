import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import normalize

class TripletLoss(nn.Module):
    def __init__(self,margin=0.8):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative,alpha,beta):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(self.margin + alpha*distance_positive - beta*distance_negative, min=0.0)
        return torch.mean(loss)


class PatchCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PatchCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        inputs: B*196
        targets: B*196
        """
        batch_size, num_patches = inputs.size()

        loss_matrix = torch.zeros(batch_size, num_patches, device=inputs.device)

        for i in range(num_patches):
            loss_fn = nn.CrossEntropyLoss()
            loss_matrix[:, i] = loss_fn(inputs[:, i].unsqueeze(1), targets[:, i].long())

        return loss_matrix

class Arc(nn.Module):
    def __init__(self, feat_num, cls_num) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn((feat_num, cls_num))) 

    def forward(self, x, m=1, s=10):
        x_norm = normalize(x, p=2, dim=1) # [N, 2]
        w_norm = normalize(self.w, p=2, dim=0) # [2, 10]
        cosa = torch.matmul(x_norm, w_norm) / s 
        a = torch.arccos(cosa) 
        top = torch.exp(s*torch.cos(a+m)) # [N, 10]
        down = top + torch.sum(torch.exp(s*cosa), dim=1, keepdim=True) - torch.exp(s*cosa)
        arc_softmax = top/(down+1e-10)
        return arc_softmax

class Instance_CE(nn.Module):
    def __init__(self):
        super(Instance_CE, self).__init__()

    def forward(self,patch_prediction , pseudo_instance_label):#pred:B*196*2,target B*196
        patch_prediction = patch_prediction.float()
        pseudo_instance_label = pseudo_instance_label.float()

        patch_prediction = torch.softmax(patch_prediction, dim=-1)

        loss_student = -1. * torch.mean(
            (1 - pseudo_instance_label) * torch.log(patch_prediction[:, :, 0] + 1e-8) +
            pseudo_instance_label * torch.log(patch_prediction[:, :, 1] + 1e-8)
        )

        return loss_student


class compute_L_NC(nn.Module):
    def __init__(self):
        super(compute_L_NC, self).__init__()
    
    def forward(self, F_nc, bag_labels, classifier_B):
        """
        Compute the NC loss.

        Args:
            F_nc: [B, 1, d]
            bag_labels: tensor of shape [B]
            classifier_B: Bag-level classifier head

        Returns:
            L_NCD: scalar tensor
        """
        Wr_labels = 1-bag_labels
        preds = classifier_B(F_nc)
        loss = F.cross_entropy(preds, Wr_labels)
        return loss



def CMD(x1, x2, k=4):
    mx1 = torch.mean(x1, 0)
    mx2 = torch.mean(x2, 0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = matchnorm(mx1, mx2)
    scms = dm
    for i in range(k - 1):
        scms += scm(sx1, sx2, i + 2)
    return scms

def matchnorm(x1, x2):
    print(x1, x2)
    power = torch.pow(x1 - x2, 2)
    summed = torch.sum(power)
    sqrt = summed ** (0.5)
    return sqrt

def scm(sx1, sx2, k):
    ss1 = torch.mean(torch.pow(sx1, k), 0)
    ss2 = torch.mean(torch.pow(sx2, k), 0)
    return matchnorm(ss1, ss2)

class compute_L_CI(nn.Module):
    def __init__(self):
        super(compute_L_CI, self).__init__()
        self.loss_mse = nn.MSELoss()

    def forward(self, F_c, F_nc, fusion_layer, classifier_B, k_moments=5):
        """
        Compute the CI loss.

        Args:
            F_c: Causal features [B, d]
            F_nc: Non-causal features [B, d] 
            fusion_layer: Fusion layer to combine F_c and F_nc
            classifier_B: Bag-level classifier head
            k_moments: Number of moments to consider (default is 5)

        Returns:
            L_CPI: scalar tensor
        """
        B, D = F_c.shape
        loss_total = 0.0

        for m in range(B):
            f_c_m = F_c[m].unsqueeze(0)  # [1, d]
            z_c = classifier_B(f_c_m)     # [1, num_classes]

            for n in range(B):
                f_nc_n = F_nc[n].unsqueeze(0)  # [1, d]
                # Fusion: concatenate and transform
                f_fused = fusion_layer(torch.cat([f_c_m, f_nc_n], dim=1))  # [1, d_fused]
                z_comb = classifier_B(f_fused)  # [1, num_classes]
                
                loss_total += self.loss_mse(z_c, z_comb)

        L_CI = loss_total / (B * B)
        return L_CI


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        # print('mean:', log_preds.shape)   8*2
        # print(target.shape)  8
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)