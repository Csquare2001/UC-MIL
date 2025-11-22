from dataset_single import dataset_npy
from torch.utils.data import DataLoader
from model.KLmodel_cc import MIL_vit as create_model
from torchnet import meter
from smooth_label import LabelSmoothingCrossEntropy#,instance_CE
from lossfunction.loss import Arc,PatchCrossEntropyLoss,TripletLoss,Instance_CE
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
import sys
import torch.optim.lr_scheduler as lr_scheduler
import math,sys
import numpy as np
import random
import os
import torch.nn.functional as F
sys.path.append("model")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda:0');device_ids = [0]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


loss_arc = Arc(196, 196).to(device)
loss_lsce = LabelSmoothingCrossEntropy()
loss_fn = PatchCrossEntropyLoss()
loss_ce = Instance_CE()
loss_triplet = TripletLoss()
loss_constritive = torch.nn.TripletMarginLoss(margin=0.8)
loss_kl = torch.nn.KLDivLoss(size_average = False)
log_sm = torch.nn.LogSoftmax(dim = 1)
weight= 1


def lst(label_train):
    label_list = []
    for _ in range(2):
        label_list.append([])
    for label, item in enumerate(label_train):
        for k in range(2):
            if item == k:
                label_list[k].append(label)
    return label_list

def positive_pair(label, lst1, label_list):
    for k in range(2):
        if label == k:
            index = random.choice(label_list[k])
            lst1.append(index)
            break
    return lst1

def negative_pair(label, lst, label_list):
    Flag = True
    while Flag:
        k = random.choice([K for K in range(2)])
        if label != k:
            index = random.choice(label_list[k])
            lst.append(index)
            Flag = False
        else:
            continue
    return lst

def cal_kl(instance_cla_block11,instance_cla_block12):
    patch_prediction_11 = torch.softmax(instance_cla_block11, dim=-1)
    patch_prediction_12 = torch.softmax(instance_cla_block12, dim=-1)
    instance_cla_block11 = patch_prediction_11[:, :, 1]
    instance_cla_block12 = patch_prediction_12[:, :, 1]
    mean_instance_pred = torch.nn.Softmax(dim=1)(0.5*instance_cla_block11 + instance_cla_block12)
    kl_divergences = torch.zeros(instance_cla_block12.size(0), instance_cla_block12.size(1)).cuda()
    for i in range(instance_cla_block12.size(0)):
        for j in range(instance_cla_block12.size(1)):
            kl_div = (loss_kl(log_sm(instance_cla_block11)[i, j], mean_instance_pred[i, j]) +
                    loss_kl(log_sm(instance_cla_block12)[i, j], mean_instance_pred[i, j])).item()
            kl_divergences[i, j] = kl_div
    min_val = kl_divergences.min()
    max_val = kl_divergences.max()
    kl_divergences = (kl_divergences - min_val) / (max_val - min_val)
    return kl_divergences,mean_instance_pred



def train():
    seed=40
    setup_seed(seed)

    # train_data_root = '/media/user/Disk02/zyl/split_600/npy_crop_600/image'
    # print("************************************************")
    # data_split_path='/media/user/Disk02/zyl/re_mil420'

    train_data_root = "1017/2024_data_all_cc"
    print("************************************************")
    data_split_path="1017"
    data_split = ['12345','23451','34512','45123','51234']
    for i in range(5):
        print("*****ROUND--{}*****".format(i))
        batchsize=4
        train_data = dataset_npy(train_data_root,data_split_path,data_split[i],is_transform=True, train=True)
        print(len(train_data))
        # train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
        model=create_model()
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()#must have decica[0],the main card
        print("model:",next(model.parameters()).device)
        model_name="600_mil_beta_arc_instance_constraloss_KLuncertainty_localagg"
        print(model_name)
        epochs=100
        lrf=0.01;lr=0.0001
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        va_acc_list = [];va_acu_list = []
        test_acc_list=[];test_acu_list=[]
        if not os.path.exists(
                './results/{}-{}-{}-{}/{}'.format(model_name,batchsize,lr,seed,data_split[i])):
            os.makedirs(
                './results/{}-{}-{}-{}/{}'.format(model_name,batchsize,lr,seed,data_split[i]))
        submit_path =  './results/{}-{}-{}-{}/{}/'.format(model_name,batchsize,lr,seed,data_split[i])
        submit_file_name = '{}-{}-{}-{}-{}.csv'.format(model_name,batchsize,lr,seed,weight)
        csv_file = open(submit_path + submit_file_name, 'w')
        csv_file.write("epoch, tr_loss, tr_auc, tr_acc, va_auc, va_acc, va_sens, va_spec,te_auc, te_acc, te_sens, te_spec\n")


        for epoch in range(epochs):
            model.train()
            train_loss_epoch,cla_loss_epoch,mil_loss_epoch,instance_loss_epoch,cla_constra_epoch,mil_constra_epoch = 0.,0.,0.,0.,0.,0.
            all_predictions = []
            if(epoch > 0):
                new_patch_labels = {}
                start = 0
                for img_path in train_data.imgs:
                    end = start + 1
                    new_patch_labels[img_path] = predictions[start:end].detach().numpy()
                    start = end
                train_data.update_labels(new_patch_labels)
            train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
            if (epoch >= 80):
                if not os.path.exists('./draw_kl_agg2/{}/{}'.format(model_name,data_split[i])):
                    os.makedirs('./draw_kl_agg2/{}/{}'.format(model_name,data_split[i]))
                submit_draw_path = './draw_kl_agg2/{}/{}'.format(model_name,data_split[i])
                submit_draw_filename = '/epoch_{}_output.txt'.format(epoch)

            for step, (data, label,img_path,patch_label) in enumerate (train_dataloader):
                cla_constra = 0
                mil_constra = 0
                positive_pairs = []
                negative_pairs = []
                positive_pairs_train = []
                negative_pairs_train = []
                label_list = list(label)

                label = torch.LongTensor(label).to(device)
                cla,mil_bag,instance_cla_pri,instance_cla_aux=model(data,patch_label)
                p = (torch.softmax(instance_cla_pri, dim=-1) + torch.softmax(instance_cla_aux, dim=-1)) / 2
                pt = p**(1/0.5)
                pseudo_label = pt / pt.sum(dim=-1, keepdim=True)
                pseudo_label = pseudo_label[:,:,1]
                pseudo_label = pseudo_label.squeeze()
                # print("pseudo_label",pseudo_label.shape)
                # pseudo_label = torch.softmax(instance_cla_pri,dim=-1)
                # pseudo_label = pseudo_label[:,:,1]

                # if epoch < 1:
                #     if(step == 0):
                #         print("pseudo_label:",pseudo_label)
                #         print("patch_label:",patch_label)
                # else:
                #     if(step == 0):
                #         print("pseudo_label:",pseudo_label)
                #         print("patch_label:",patch_label)

                ########################### KL Uncertainty ##########################################
                kl_divergences,_ = cal_kl(instance_cla_pri,instance_cla_aux)
                #####################################################################################

                ########################## bag contrastive learning #################################
                if all(len(lst) > 0 for lst in label_list):
                    for ij in range(len(data)):
                        positive_pairs = positive_pair(label[ij], positive_pairs, label_list)
                        negative_pairs = negative_pair(label[ij], negative_pairs, label_list)

                    for _, label_id in enumerate(positive_pairs):
                        x_temp = train_data[label_id]
                        positive_pairs_train.append(x_temp)

                    for _, label_id in enumerate(negative_pairs):
                        x_temp1 = train_data[label_id]
                        negative_pairs_train.append(x_temp1)

                    data_p = positive_pairs_train[0][0].reshape(-1, 196, 16, 16)
                    data_n = negative_pairs_train[0][0].reshape(-1, 196, 16, 16)

                    for ii in range(len(label)-1):
                        data_p = torch.cat([data_p,(positive_pairs_train[ii+1][0]).reshape(-1,196, 16, 16)],0)

                    for ii in range(len(label)-1):
                        data_n = torch.cat([data_n,(negative_pairs_train[ii+1][0]).reshape(-1,196,16, 16)],0)

                    data_p = data_p.to(device)
                    data_n = data_n.to(device)


                    cla_constra_0,mil_constra_0,instance_cla_pri_constra,instance_cla_aux_constra = model(data_p,patch_label)
                    cla_constra_1,mil_constra_1,instance_cla_pri_constra1,instance_cla_aux_constra1 = model(data_n,patch_label)

                    cla_constra = loss_constritive(cla,cla_constra_0,cla_constra_1)
                    # mil_constra = loss_constritive(mil,mil_constra,mil_constra_1)
                #################################################################################################

                #########uncertainty loss#########################################################################
                # out = loss_arc(instance_cla_pri.squeeze())
                # Lce = loss_fn(out,pseudo_label)
                # print(label.size())
                # print(pseudo_label.size())

                # Lce = loss_fn(instance_cla_pri,torch.LongTensor(pseudo_label).to(device))
                # print(instance_cla_pri.size())
                # print(pseudo_label.size())
                all_predictions.append(pseudo_label.cpu())
                Lce = loss_ce(instance_cla_pri,patch_label.to(device))
                loss_instance = torch.mean(torch.exp(-kl_divergences)*Lce + kl_divergences)
                # loss_instance =torch.nn.functional.binary_cross_entropy_with_logits(out*kl_divergences,pseudo_label)
                #################################################################################################
                loss_cla= loss_lsce(cla, label)
                loss_mil = loss_lsce(mil_bag,label)
                #loss_instance = instance_CE(insatance_cla,pseudo_label)

                loss = loss_cla +loss_mil + loss_instance+cla_constra
                optimizer.zero_grad()
                # torch.autograd.detect_anomaly(True)
                loss.backward()
                optimizer.step()

                # train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
                train_loss_epoch+=loss.item();cla_loss_epoch+=loss_cla.item();mil_loss_epoch+=loss_mil.item();instance_loss_epoch+=loss_instance.item();cla_constra_epoch+=cla_constra;mil_constra_epoch+=mil_constra

                if (epoch >= 80):
                    with open(submit_draw_path+submit_draw_filename, 'a') as f:
                        for index in range(len(data)):
                            f.write(f"Training data: {img_path[index]}\n")
                            f.write(f"label matrix: {patch_label[index]}\n")
                            save_cla = torch.softmax(instance_cla_pri, dim=-1)
                            save_cla = save_cla[:,:,1]
                            f.write(f"preds matrix: {save_cla[index]}\n")
                            f.write(f"uncertainty matrix:{kl_divergences[index]}\n")
                            f.write(f"----------------------------------------------------------------------\n")
            predictions = torch.cat(all_predictions)
            # new_patch_labels = predictions
            print(scheduler.get_last_lr())
            print("train_loss: {:.3f}   cla_loss: {:.3f}   mil_loss: {:.3f}   instance_loss: {:.3f}  cla_constra:{:.3f} mil_constra:{:.3f}".
                  format(train_loss_epoch/batchsize, cla_loss_epoch/batchsize, mil_loss_epoch/batchsize, instance_loss_epoch/batchsize,cla_constra_epoch/batchsize,mil_constra_epoch/batchsize))
            scheduler.step()

            pat_train_data  = dataset_npy(train_data_root, data_split_path, data_split[i], is_transform=False, train=True)
            pat_val_data = dataset_npy(train_data_root, data_split_path, data_split[i],is_transform=False, val=True)
            pat_test_data = dataset_npy(train_data_root, data_split_path, data_split[i], is_transform=False, test=True)

            tr_auc, tr_acc, tr_sens, tr_spec, tr_los,tr_score,tr_lab,tr_false = val(model, pat_train_data)
            va_auc, va_acc, va_sens, va_spec, va_los,va_score,va_lab,va_false = val(model, pat_val_data)
            # te_auc, te_acc, te_sens, te_spec, te_los, te_score, te_lab,te_false = val(model, pat_test_data)

            va_acc_list.append(va_acc);va_acu_list.append(va_auc)
            # test_acc_list.append(te_acc);test_acu_list.append(te_auc)

            print('******************Epoch: ', epoch, '******************')
            print("train_loss: {:.3f}, train_auc:{:.3f},train_acc: {:.3f}".format(tr_los, tr_auc, tr_acc))
            print("val_loss: {:.3f}, val_auc:{:.3f},val_acc: {:.3f}".format( va_los,va_auc,va_acc ))
            # print("te_loss: {:.3f}, te_auc:{:.3f},te_acc: {:.3f}".format(te_los, te_auc, te_acc))

            if not os.path.exists(
                    submit_path + 'va'):
                os.makedirs(
                      submit_path + 'va')
            if not os.path.exists(
                      submit_path + 'te'):
                os.makedirs(
                    submit_path + 'te')
            # np.savetxt(submit_path + 'te/{}-{}-{}-{}te_score.csv'.format(batchsize, lr, seed, epoch),
            #                 np.concatenate([te_score, te_lab], axis=1), delimiter=',')
            np.savetxt(submit_path + 'va/{}-{}-{}-{}va_score.csv'.format(batchsize, lr, seed, epoch),
                       np.concatenate([va_score, va_lab], axis=1), delimiter=',')
            csv_file.write(
                str(epoch) + ',' + str(round(loss.item(), 3)) + ',' + str(tr_auc) + ',' + str(tr_acc) + ','
                + str(va_auc) + ',' + str(va_acc) + ',' + str(va_sens) + ',' + str(va_spec) + ',')
                 # +str(te_auc) + ',' + str(te_acc) + ',' + str(te_sens) + ',' + str(te_spec)+'\n')
        csv_file.write('va_acc_max' + ',' + str(max(va_acc_list)) + ',' + 'va_auc_max' + ',' + str(max(va_acu_list))+','+
            'te_acc_max' + ',' + str(max(test_acc_list)) + ',' + 'te_auc_max' + ',' + str(max(test_acu_list)))
        print("max_va_acc:{:.3f},max_va_auc:{:.3f},max_te_acc:{:.3f},max_te_auc:{:.3f}".
              format(max(va_acc_list),max(va_acu_list),max(test_acc_list),max(test_acu_list)))


@torch.no_grad()
def val(model, data):

    dataloader = DataLoader(data, 3, shuffle=False, num_workers=2)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    loss_av = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    false_name = []

    score = np.array([]).reshape(0, 1)
    lab = np.array([]).reshape(0, 1)
    count = 0
    # score = []
    # lab   = []

    for ii, (input, label,_,_) in enumerate(dataloader):
        input = input.to(device)
        label3 = label.to(device)

        cla_score= model(input,label3)
        '''cla_score, rec_score, fea = model(input)
        cla_score2, rec_score2, fea2 = model(rec_score)'''

        # pro = t.nn.functional.softmax\
        # (cla_score)[:,1].data.tolist()

        pro = torch.nn.functional.softmax(cla_score,dim=1)
        cla = torch.mean(pro, dim=0).view(1, 2).detach()  # tensor([[0.5952, 0.4048]])
        _, pred = cla.max(1)  # predict result
        if (pred == label[0]).item() != True:
            # print(data.imgs[i * 3].split('/')[-1].split('.')[-2] + "false predicition")
            false_name.append(data.imgs[ii * 3].split('/')[-1].split('.')[-2].split('_')[-2])

        # fc/softmax layer output shape : (batch_size, 2)
        # for calculate auc
        cla_np = np.mean(pro[:, 1].cuda().data.cpu().numpy()).reshape(-1, 1) 
        lab_np = label3.cuda().data.cpu().numpy()[0:1].reshape(-1, 1)

        score = np.concatenate([score, cla_np], 0)
        lab = np.concatenate([lab, lab_np], 0)

        confusion_matrix.add(torch.mean(pro, dim=0).view(1, 2).detach(), label[0:1].type(torch.LongTensor))

        loss = loss_lsce(cla_score, label3)
        loss_av.add(loss.item())

    model.train()
    false_name = np.array(false_name)
    cm_value = confusion_matrix.value()
    # print(cm_value)
    accuracy =  (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum()+1e-8)
    sens_c = cm_value[0][0] / (cm_value[0][0] + cm_value[0][1]+1e-8)
    spec_c = cm_value[1][1] / (cm_value[1][1] + cm_value[1][0]+1e-8)
    if np.isnan(score).any() or np.isinf(score).any():
        score = np.nan_to_num(score)
    AUC = roc_auc_score(lab, score)

    # cla.value() --> (mean,std)
    return AUC, accuracy, sens_c, spec_c, loss_av.value()[0],score,lab,false_name



if __name__=="__main__":
    train()

