from tqdm import tqdm

from dataset_evi_cau import dataset_npy
from torch.utils.data import DataLoader
from model.model_evidence_causal import MIL_vit as create_model
from torchnet import meter
from utils import PatchCrossEntropyLoss,TripletLoss,Instance_CE, compute_L_CI, compute_L_NC, LabelSmoothingCrossEntropy
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
device = torch.device('cuda:0'); device_ids = [0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


loss_lsce = LabelSmoothingCrossEntropy()
loss_fn = PatchCrossEntropyLoss()
loss_ce = Instance_CE()
loss_triplet = TripletLoss()
loss_constritive = torch.nn.TripletMarginLoss(margin=0.8)
loss_kl = torch.nn.KLDivLoss(size_average = False)
log_sm = torch.nn.LogSoftmax(dim = 1)
adversarial_loss = torch.nn.BCELoss()
loss_mse = torch.nn.MSELoss()
loss_NC = compute_L_NC()
loss_CI = compute_L_CI()
weight= 1



def train():
    torch.autograd.set_detect_anomaly(True)
    train_data_root = "data_1017/data_all"
    print("************************************************")
    data_split_path="/data4/caochi/UC-MIL/data_1017/split_sel_1072_203_1"
    data_split = ['12345','23451','34512','45123','51234']
    for i in range(0, 5):
        print("*****ROUND--{}*****".format(i))
        batchsize = 8
        seed = 42
        setup_seed(seed)
        train_data = dataset_npy(train_data_root,data_split_path,data_split[i],is_transform=True, train=True)
        print(len(train_data))
        model=create_model()
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()#must have decica[0],the main card
        print("model:",next(model.parameters()).device)
        model_name="1016_mil_Evi_uncertainty_Casuality_fold5-4-3-2-1(cc2_final)_te_pub"
        print(model_name)
        epochs=100
        lrf=0.01; lr = 0.0001
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        va_acc_list = [];va_acu_list = []
        test_acc_list=[];test_acu_list=[]
        if not os.path.exists(
                './results_xk_xxx/fd-cau/{}-{}-{}-{}/{}'.format(model_name,batchsize,lr,seed,data_split[i])):
            os.makedirs(
                './results_xk_xxx/fd-cau/{}-{}-{}-{}/{}'.format(model_name,batchsize,lr,seed,data_split[i]))
        submit_path =  './results_xk_xxx/fd-cau/{}-{}-{}-{}/{}/'.format(model_name,batchsize,lr,seed,data_split[i])
        submit_file_name = '{}-{}-{}-{}-{}.csv'.format(model_name,batchsize,lr,seed,weight)
        csv_file = open(submit_path + submit_file_name, 'w')
        csv_file.write("epoch, va_auc, va_acc, va_sens, va_spec\n")


        for epoch in range(epochs):
            model.train()
            train_loss_epoch,cla_loss_epoch,mil_loss_epoch,instance_loss_epoch,cla_constra_epoch,mil_constra_epoch = 0.,0.,0.,0.,0.,0.
            all_predictions = []
            if(epoch > 10):
                new_patch_labels = {}
                start = 0
                for img_path in train_data.imgs:
                    end = start + 1
                    new_patch_labels[img_path] = predictions[start:end].detach().numpy()
                    start = end
                train_data.update_labels(new_patch_labels)
                # print(predictions.shape)          # 1593*196

            train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
            if (epoch >= 80):
                if not os.path.exists('./draw_kl_agg2/{}/{}'.format(model_name,data_split[i])):
                    os.makedirs('./draw_kl_agg2/{}/{}'.format(model_name,data_split[i]))
                submit_draw_path = './draw_kl_agg2/{}/{}'.format(model_name,data_split[i])
                submit_draw_filename = '/epoch_{}_output.txt'.format(epoch)

            for step, (data, label, img_path, patch_label, _) in enumerate(tqdm(train_dataloader)):

                # print(data.shape)
                cla_constra = 0
                mil_constra = 0

                label = torch.LongTensor(label).to(device)

                ### model output
                instance_pred, uncertainty, CauScore, topk_indices, F_c, F_nc, bag_pred = model(data)
                # print(instance_pred.shape)
                instance_pred = torch.softmax(instance_pred, dim=-1)
                pseudo_label = instance_pred[:, :, 1]

                all_predictions.append(pseudo_label.cpu())
                Lce = loss_ce(instance_pred[:, :, 1], patch_label.to(device))
                loss_instance = torch.mean(torch.exp(-uncertainty)*Lce)

                #################################################################################################
                loss_bag = loss_lsce(bag_pred, label)
                # loss_cls = loss_lsce(cls_token, label)

                # print(F_nc.shape, label.shape)
                loss_nc = loss_NC(F_nc, label, model.module.bag_classification)
                loss_ci = loss_CI(F_c, F_nc, model.module.fusion_layer, model.module.bag_classification)

                loss = loss_bag + 0.8 * loss_instance + 0.5 * loss_nc + 0.6 * loss_ci

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                # train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
                train_loss_epoch+=loss.item();mil_loss_epoch+=loss_bag.item();instance_loss_epoch+=loss_instance.item();cla_constra_epoch+=cla_constra;mil_constra_epoch+=mil_constra

                if (epoch >= 80):
                    with open(submit_draw_path+submit_draw_filename, 'a') as f:
                        for index in range(len(data)):
                            f.write(f"Training data: {img_path[index]}\n")
                            f.write(f"label matrix: {patch_label[index]}\n")
                            f.write(f"preds matrix: {pseudo_label[index]}\n")
                            f.write(f"uncertainty matrix:{uncertainty[index]}\n")
                            f.write(f"----------------------------------------------------------------------\n")
            predictions = torch.cat(all_predictions)
            # new_patch_labels = predictions

            print('\n当前学习率')
            print(scheduler.get_last_lr())
            print("train_loss: {:.3f}   cla_loss: {:.3f}   mil_loss: {:.3f}   instance_loss: {:.3f}  cla_constra:{:.3f} mil_constra:{:.3f}".
                  format(train_loss_epoch/batchsize, cla_loss_epoch/batchsize, mil_loss_epoch/batchsize, instance_loss_epoch/batchsize,cla_constra_epoch/batchsize,mil_constra_epoch/batchsize))
            scheduler.step()


            # pat_train_data  = dataset_npy(train_data_root, data_split_path, data_split[i], is_transform=False, train=True)
            pat_val_data = dataset_npy(train_data_root, data_split_path, data_split[i], is_transform=False, val=True)        # 612
            pat_fy_data = dataset_npy(train_data_root, data_split_path, data_split[i], is_transform=False, test3=True)
            pat_pub_data = dataset_npy(train_data_root, data_split_path, data_split[i], is_transform=False, test1=True)

            # tr_auc, tr_acc, tr_sens, tr_spec, tr_los,tr_score,tr_lab,tr_false, tr_name = val(model, pat_train_data)
            va_auc, va_acc, va_sens, va_spec, va_los, va_score, va_lab, va_false, va_name = val(model, pat_val_data)
            fy_auc, fy_acc, fy_sens, fy_spec, fy_los, fy_score, fy_lab, fy_false, fy_name = val(model, pat_fy_data)
            pub_auc, pub_acc, pub_sens, pub_spec, pub_los, pub_score, pub_lab, pub_false, pub_name = val(model, pat_pub_data)


            va_acc_list.append(va_acc); va_acu_list.append(va_auc)
            print("-----------------", va_auc.shape)

            print('******************Epoch: ', epoch, '******************')
            # print("train_loss: {:.3f}, train_auc:{:.3f},train_acc: {:.3f}".format(tr_los, tr_auc, tr_acc))
            print("val_loss: {:.3f}, val_auc:{:.3f},val_acc: {:.3f}".format( va_los,va_auc,va_acc ))
            # print("te_loss: {:.3f}, te_auc:{:.3f},te_acc: {:.3f}".format(te_los, te_auc, te_acc))

            if not os.path.exists(
                    submit_path + 'va'):
                os.makedirs(
                      submit_path + 'va')

            np.savetxt(submit_path + 'va/{}-{}-{}-{}va_score.csv'.format(batchsize, lr, seed, epoch),
                       np.concatenate([va_name, va_score, va_lab], axis=1), fmt="%s", delimiter=',',
                       header="name,score,label", comments='')



            if not os.path.exists(
                    submit_path + 'fy'):
                os.makedirs(
                      submit_path + 'fy')

            np.savetxt(submit_path + 'fy/{}-{}-{}-{}fy_score.csv'.format(batchsize, lr, seed, epoch),
                       np.concatenate([fy_name, fy_score, fy_lab], axis=1), fmt="%s", delimiter=',',
                       header="name,score,label", comments='')



            if not os.path.exists(
                    submit_path + 'pub'):
                os.makedirs(
                      submit_path + 'pub')

            np.savetxt(submit_path + 'pub/{}-{}-{}-{}pub_score.csv'.format(batchsize, lr, seed, epoch),
                       np.concatenate([pub_name, pub_score, pub_lab], axis=1), fmt="%s", delimiter=',',
                       header="name,score,label", comments='')



            csv_file.write(
                str(epoch) + ',' + str(va_auc) + ',' + str(va_acc) + ',' + str(va_sens) + ',' + str(va_spec) + ',' + \
                str(fy_auc) + ',' + str(fy_acc) + ',' + str(fy_sens) + ',' + str(fy_spec) + ',' + \
                str(pub_auc) + ',' + str(pub_acc) + ',' + str(pub_sens) + ',' + str(pub_spec) + '\n')





@torch.no_grad()
def val(model, data):
    """
    计算模型在验证集上的准确率等信息
    return: 分割损失，重分割损失，重建损失，一致性损失
    """

    dataloader = DataLoader(data, 3, shuffle=False, num_workers=2)
    print(len(data))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    loss_av = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    false_name = []
    score = np.array([]).reshape(0, 1)
    lab = np.array([]).reshape(0, 1)
    name = np.array([]).reshape(0, 1)


    for ii, (input, label, _, patch_label, name1) in enumerate(dataloader):
        input = input.to(device)
        label3 = label.to(device)
        patch_label = patch_label.to(device)
        name1 = np.array(list(name1))

        instance_pred, uncertainty, CauScore, topk_indices, F_c, F_nc, bag_pred = model(input)

        pro = torch.nn.functional.softmax(bag_pred, dim=1)
        cla = torch.mean(pro, dim=0).view(1, 2).detach()  # tensor([[0.5952, 0.4048]])
        _, pred = cla.max(1)  # predict result
        if (pred == label[0]).item() != True:
            false_name.append(data.imgs[ii * 3].split('/')[-1].split('.')[-2].split('_')[-2])

        cla_np = np.mean(pro[:, 1].cuda().data.cpu().numpy()).reshape(-1, 1)

        lab_np = label3.cuda().data.cpu().numpy()[0:1].reshape(-1, 1)
        name_t = name1.reshape(-1, 1)[0:1]

        score = np.concatenate([score, cla_np], 0)
        lab = np.concatenate([lab, lab_np], 0)
        name = np.concatenate([name, name_t], 0)

        confusion_matrix.add(torch.mean(pro, dim=0).view(1, 2).detach(), label[0:1].type(torch.LongTensor))

        loss = loss_lsce(bag_pred, label3)
        loss_av.add(loss.item())

    model.train()
    false_name = np.array(false_name)
    cm_value = confusion_matrix.value()
    accuracy =  (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum()+1e-8)
    sens_c = cm_value[0][0] / (cm_value[0][0] + cm_value[0][1]+1e-8)
    spec_c = cm_value[1][1] / (cm_value[1][1] + cm_value[1][0]+1e-8)
    if np.isnan(score).any() or np.isinf(score).any():
        score = np.nan_to_num(score)
    AUC = roc_auc_score(lab, score)
    print(score.shape, name.shape)
    return AUC, accuracy, sens_c, spec_c, loss_av.value()[0],score,lab, false_name, name



if __name__=="__main__":
    train()
