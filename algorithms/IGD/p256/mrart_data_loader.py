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


class MRART(data.Dataset):

    # constructor of the class
    def __init__(self, root: str, normal_class,
                task, seed,N, transform, data_split_path):
        self.data = []
        self.targets=[]
        self.root_dir = root
        self.task = task
        self.transform = transform
        self.targets_sev=[]
        self.original_files=[]

        scores = pd.read_csv(root+ '/scores.tsv', sep='\t')
        scores.columns = ['bids_name', 'score']
        random.seed(seed)
        normal_class =1
        normals = scores.loc[scores['score']==1]['bids_name'].reset_index(drop=True)
        idx= random.sample(list(range(0, len(normals))),N)
        train_files =normals[idx].reset_index(drop=True)

        ids = train_files.apply(lambda x: x.split('_')[0]).tolist()
        self.indexes = train_files.values.tolist()
        #check its correct training data
        check = pd.read_csv(data_split_path + 'df_mrart_seed_' + str(seed) + '_n_' +str(N))

        for f in ids:
            assert f in list(check['file'].loc[check['split'] =='train'])



        if task =='train':


                path = root +  '/ones/'

                for file in train_files:
                    for slice in range(192):
                        file_name = path + file + '_slice_' + str(slice) +'.png'
                        im = cv2.imread(file_name)

                        im = np.stack((im[:,:,0],im[:,:,1],im[:,:,2]))
                        self.data.append(im)

                        self.targets.append(0)
                        self.targets_sev.append(0)
                        self.original_files.append(file)



        else:
            c = 0
            paths = [root +'/ones/', root +'/twos/', root +'/threes/']
            for i,path in enumerate(paths):
                files = os.listdir(path)
                for file in files:
                    if file.split('_')[0] not in ids:
                            file_name = path + file #+ '_slice_' + str(slice)
                            im = cv2.imread(file_name)
                        #    im2 =cv2.resize( im[:,:,0] , (64,64))
                        #    im3 =cv2.resize( im[:,:,1] , (64,64))
                        #    im4 =cv2.resize( im[:,:,2] , (64,64))
                            im = np.stack((im[:,:,0],im[:,:,1],im[:,:,2]))
                            self.data.append(im)

                            if i ==0:
                                self.targets.append(0)
                            else:
                                self.targets.append(1)

                            c+=1

                            if c % 10000 == 0:
                                print(c)

                            ind = file.split('_')[-1]
                            ind = int(re.search(r'\d+', ind).group())
                            if ind <10:
                                self.original_files.append(file[:-12])
                            elif ind <100:
                                self.original_files.append(file[:-13])
                            else:
                                self.original_files.append(file[:-14])

                            if i == 0:
                                self.targets_sev.append(0)
                            elif i ==1:
                               self.targets_sev.append(1)
                            elif i ==2:
                                self.targets_sev.append(2)
            for f in self.original_files:
                assert (f.split('_')[0] + '_' +  f.split('_')[1] )  in list(check['file'].loc[check['split'] =='validation'])
                assert (f.split('_')[0] )   not in list(check['file'].loc[check['split'] =='train'])






    #    self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)

        label = target
        img = transforms.Normalize([77.3,77.3,77.3], [140,140, 140])(img)
        return img, label, int(self.targets_sev[index]), self.original_files[index]

    def __len__(self):
        return len(self.data)
