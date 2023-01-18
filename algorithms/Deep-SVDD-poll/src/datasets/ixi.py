import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import cv2
import random
import pandas as pd
import re

import torchvision.transforms as transforms

class IXI_Dataset(data.Dataset):


    training_file = 'training.pt'
    test_file = 'test.pt'


    def __init__(self, root, pollution, N, normal_class, task, seed, data_split_path
    ) -> None:
        super().__init__()
        self.task = task  # training set or test set
        self.data_path = root
        self.normal_class = normal_class
        self.task = task

        self.data = []
        self.targets=[]
        self.root_dir = root
        self.task = task
        self.targets_sev=[]
        self.original_files=[]
        self.indexes=[]



        normals = os.listdir(root+ '/normal/')
        normals_ids = pd.DataFrame(normals.copy()).iloc[:,0].apply(lambda x: x.split('_')[2])
        normals_ids.columns=['file']
        normals_ids=normals_ids.drop_duplicates().reset_index(drop=True)#.apply(lambda x: x[0]+'_'+x[1])

        random.seed(seed)
        ind = random.sample(range(0, len(normals_ids)), N)
        train_files = np.array(normals_ids)[ind] #names of training


        check = pd.read_csv(data_split_path + 'df_seed_' + str(seed) + '_n_' +str(N))
        for f in train_files:
            assert f in list(check['file'].loc[check['split'] =='train'])



        if task =='train':
            for i,f in enumerate(train_files):
            #    print('f is {}'.format(f))
                path_temp=[]
                for file in normals:
                    if  f in file.split('_')[2] : #
                #        print('f: {} is in {}'.format(f, file))
                        im = cv2.imread(root+ '/normal/' + file)
                        im2 = cv2.resize(im[:,:,0], (64,64))
                        im3 = cv2.resize(im[:,:,1], (64,64))
                        im4 = cv2.resize(im[:,:,2], (64,64))
                        im = np.stack((im2,im3,im4),0)
                        self.data.append(im)
                        self.targets.append(0)
                        self.targets_sev.append(0)
                        self.original_files.append(f+'_norm')
                        self.indexes.append(i)


        else:
            val = os.listdir(root + '/anom/')
            val_files = pd.DataFrame(val).iloc[:,0].apply(lambda x: x.split('_')[2]).drop_duplicates().reset_index(drop=True)
            for f in val_files:
            #    if f =='IXI119-Guys-0765-T2':
            #        print('its in anomaly validation')
                if f not in train_files:

                    for file in val:
                        if f in file: #
                            im = cv2.imread(root+ '/anom/' + file)
                            im2 = cv2.resize(im[:,:,0], (64,64))
                            im3 = cv2.resize(im[:,:,1], (64,64))
                            im4 = cv2.resize(im[:,:,2], (64,64))
                            im = np.stack((im2,im3,im4),0)
                            #im = np.stack((im[:,:,0],im[:,:,1],im[:,:,2]))
                            self.data.append(im)
                            self.targets.append(1)
                            self.targets_sev.append(0)
                            self.original_files.append(f+'_anom')



            val = os.listdir(root + '/normal/')
            val_files = pd.DataFrame(val).iloc[:,0].apply(lambda x: x.split('_')[2]).drop_duplicates().reset_index(drop=True)
            for f in val_files:
            #    if f =='IXI119-Guys-0765-T2':
                #    print('its in nromal validation')
                if f not in train_files:
                    for file in val:
                        if f in file: #getting the files in the directory with the ID (f)
                            im = cv2.imread(root+ '/normal/' + file)
                            im2 = cv2.resize(im[:,:,0], (64,64))
                            im3 = cv2.resize(im[:,:,1], (64,64))
                            im4 = cv2.resize(im[:,:,2], (64,64))
                            im = np.stack((im2,im3,im4),0)
                            self.data.append(im)
                            self.targets.append(0)
                            self.targets_sev.append(0)
                            self.original_files.append(f+'_norm')


            for f in self.original_files:
            #    print(f.split('_')[0] )
                assert f.split('_')[0] in list(check['file'].loc[check['split'] =='validation'])
                assert f.split('_')[0]  not in list(check['file'].loc[check['split'] =='train'])




    #    self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        print('Class balance in validation is {}'.format(pd.DataFrame(self.targets).iloc[:,0].value_counts()))







    def __getitem__(self, index: int):



        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)

        label = target

        img = transforms.Normalize([55.1175, 55.1175, 55.1175], [45.3279, 45.3279, 45.3279])(img)
        return img, label, index, int(self.targets_sev[index]), self.original_files[index]



    def __len__(self) -> int:
        return len(self.data)
