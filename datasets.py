import os
import numpy as np
import pandas as pd
from torchvision import transforms
import torch as t
from torch.utils import data
from torch.utils.data import DataLoader

import random


path1 = '1017/'
random.seed(41)


class dataset_npy(data.Dataset):

    def __init__(self, root, data_split_path,data_split,is_transform=None, train=False, val=False, test=False):
        self.transforms = is_transform
        self.test = test
        self.train = train
        self.val = val


        imgs = [os.path.join(root, img) for img in os.listdir(root)]


        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[-2]))
        imgs = sorted(imgs, key=lambda x: x.split('/')[-1].split('_')[-2].split('-')[-1])
        train_keys = list(np.load(data_split_path + '/fold_' + data_split[0] + '.npy')) + \
                     list(np.load(data_split_path + '/fold_' + data_split[1] + '.npy')) + \
                     list(np.load(data_split_path + '/fold_' + data_split[2] + '.npy')) + \
                     list(np.load(data_split_path + '/fold_' + data_split[3] + '.npy'))
        val_keys = list(np.load(data_split_path + '/fold_' + data_split[4] + '.npy'))
        # test_keys = list(np.load(test_path  + '/140.npy',allow_pickle=True))


        clinical_data = np.array(pd.read_csv(path1 + 'label_1167.csv', encoding="GB2312"))

        pat_name = clinical_data[:, 0].tolist()
        label = clinical_data[:, 1].tolist() 


        if self.val:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in val_keys]
            #print(len(self.imgs))#56*3=168
            self.labels = [label[pat_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])] for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in val_keys]
        # elif self.test:
        #     self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in test_keys]
        #     self.labels = [label[pat_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])] for img in
        #                    imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in test_keys]
        else:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in train_keys]
            self.labels = [label[pat_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])] for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in train_keys]
        # self.patch_labels = np.zeros((len(self.imgs), 14 * 14), dtype=int)
        self.patch_labels = {img_path: np.zeros((14 * 14), dtype=int) for img_path in self.imgs}
        # self.patches, self.patch_labels = self._create_patches_and_labels()

    def _create_patches_and_labels(self,data,label):

        # img_path = self.imgs[index]
        # label = self.labels[index]

        patch_labels = []
        patch_num = 14
        patch_size = 224 // patch_num
        patch_data = np.zeros((1, patch_size, patch_size))  
        for i in range(patch_num):
            for j in range(patch_num):
                if (i == patch_num - 1 and j == patch_num - 1):
                    patch_img = data[:, patch_size * i:, patch_size * j:]
                    patch_data = np.concatenate([patch_data, patch_img], axis=0)
                    patch_labels.append(int(label))
                    #print(i, j, patch_img.shape)
                else:
                    patch_img = data[:, patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1)]
                    #print(i, j, patch_img.shape)
                    patch_data = np.concatenate([patch_data, patch_img], axis=0)
                    patch_labels.append(int(label))
        patch_labels = np.array(patch_labels)
        patch_data = np.delete(patch_data, obj=0, axis=0).astype('float32')
        patch_data = t.FloatTensor(patch_data)
        return  patch_data,patch_labels

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]

        data = np.load(img_path)
        data = np.clip(data, -1000, 400)
        data = (data + 1000) / (1000 + 400)
        data = data[np.newaxis, ...] # add a dim eg.(224,) --> (1,224)
        patch_data,  patch_labels = self._create_patches_and_labels(data,label)
        # self.patch_labels[index] = patch_labels

        if img_path in self.patch_labels:
            patch_labels = self.patch_labels[img_path]
        else:
            self.patch_labels[img_path] = patch_labels

        return patch_data,int(label),img_path,patch_labels

    def __len__(self):
        return len(self.imgs)

    def update_labels(self, new_patch_labels):
        for img_path, labels in new_patch_labels.items():
            self.patch_labels[img_path] = labels
        # print("Updating patch labels...")
        # self.patch_labels = new_patch_labels
        # print("New patch labels set.")

if __name__ == '__main__':
    train_data_root = '/npy_crop/image'
    data_split_path='/'
    data_split='12345'
    print("************************************************")

    train_data = dataset_npy(train_data_root,data_split_path,data_split,is_transform=True, test=True)
    print(len(train_data))
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=2)
    for epoch in range(4):
        # if(epoch > 0):
            # train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=2)
            # print("new_label:",train_data.patch_labels)
        for i, (val_input,label,img_path,patch_label)in enumerate(train_dataloader):
            if(i==5):
                print("i:",i)
                print("label:",label)
                print("input_img_size:",val_input.size())
                print("input_img_path:",img_path)
                print("patch_label_size:",patch_label.size())
                if(epoch < 1):
                    print("patch_label:",patch_label)
                else:
                    print("new_label:",train_data.patch_labels)
                new_labels = patch_label+i
                train_data.update_labels(new_labels)

                # print(train_data.imgs[i*3].split('/')[-1].split('.')[-2])
                # print(train_data.labels[i*3],label,(label[0]==train_data.labels[i*3]).item())
                # print(val_input.shape,type(val_input) )