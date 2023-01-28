import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import random
import pandas as pd
import nibabel as nib
import cv2
import re

class MRART_vote(data.Dataset):



    def __init__(self, indexes, root: str, normal_class,
            task, data_path, seed,N, sample, data_split_path):
        super().__init__()
        self.data = []
        self.targets=[]
        self.root_dir = root
        self.task = task
        self.targets_sev=[]
        self.original_files=[]

        scores = pd.read_csv(root+ '/scores.tsv', sep='\t')
        scores.columns = ['bids_name', 'score']


        normal_class =1
        normals = scores.loc[scores['score']==1]['bids_name'].reset_index(drop=True)
        random.seed(seed)
        idx= random.sample(list(range(0, len(normals))),N)
        train_files =normals[idx].reset_index(drop=True)
        print(train_files)
        ids = train_files.apply(lambda x: x.split('_')[0]).tolist()
        print(ids)
        self.indexes = train_files.values.tolist()


        #check its correct training data
        check = pd.read_csv(data_split_path + 'df_mrart_seed_' + str(seed) + '_n_' +str(N))
        print(seed)
        print(N)
        print(list(check['file'].loc[check['split'] =='train']))
        for f in ids:
            print(f)
            assert f in list(check['file'].loc[check['split'] =='train'])





        if task =='train':


                path = root +  '/separate/ones/'

                for file in train_files:
                        slice = 95
                    #for slice in range(192):
                        file_name = path + file + '_slice_' + str(slice) +'.png'
                        im = cv2.imread(file_name)

                        im = np.stack((im[:,:,0],im[:,:,1],im[:,:,2]))
                        self.data.append(im)

                        self.targets.append(0)
                        self.targets_sev.append(0)
                        self.original_files.append(file)



        else:
            c = 0
            paths = [root +'/separate/ones/', root +'/separate/twos/', root +'/separate/threes/']
            for i,path in enumerate(paths):
                files = os.listdir(path)
                for file in files:
                    if file.split('_')[0] not in ids:
                        if int(file.split('_')[-1][:-4]) == 95:
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

        self.targets = np.array(self.targets)

        print(len(self.targets))
        print(len(self.data))






    #    self.data = np.array(self.data)




    def __len__(self):
        return len(self.data)



    def __getitem__(self, index: int, seed = 1, base_ind=-1):



        base = False
        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)
        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes) )
            c=1
            while (ind == index):
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes) )
                c=c+1

            if ind == base_ind:
              base = True

            img2 = self.data[ind]
            img2 = torch.FloatTensor(img2)
            label = torch.FloatTensor([0])

        else:
            img2 = torch.Tensor([1])
            label = target

        return img, img2, label, base, int(self.targets_sev[index]), self.original_files[index]
