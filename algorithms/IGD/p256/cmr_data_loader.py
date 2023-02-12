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


class CMR(data.Dataset):

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

        scores = pd.read_csv(root+ '/IQA.csv')
        scores.columns = ['Image', 'score']
        random.seed(seed)
        normal_class =1
        normals = scores.loc[scores['score']==1]['Image'].reset_index(drop=True)
        idx= random.sample(list(range(0, len(normals))),N)
        train_files =normals[idx].reset_index(drop=True)

        ids = train_files.apply(lambda x: x.split('-')[0]).tolist()
        self.indexes = train_files.values.tolist()
        #check its correct training data
        check = pd.read_csv(data_split_path + 'df_seed_' + str(seed) + '_n_' +str(N))

        for f in ids:
            assert f in list(check['file'].loc[check['split'] =='train'])


        print(ids)
        if task =='train':


                path = root +  '/ones/'

                for file in train_files:

                    for slice in range(11):
                        file_name = path + file + '_slice_' + str(slice) +'.png'
                        im = cv2.imread(file_name)

                        try:
                            im2 =cv2.resize( im[:,:,0] , (256,256))
                            im3 =cv2.resize( im[:,:,1] , (256,256))
                            im4 =cv2.resize( im[:,:,2] , (256,256))
                            im = np.stack((im2,im3,im4))
                            self.data.append(im)

                            self.targets.append(0)
                            self.targets_sev.append(0)
                            self.original_files.append(file)
                        except:
                            print('')


                print('N is {}'.format(N))
                print('Lenght of data is {}'.format(len(self.data)))




        else:
            c = 0
            paths = [root +'/ones/', root +'/twos/', root +'/threes/']
            for i,path in enumerate(paths):
                files = os.listdir(path)
                for file in files:
                    if file.split('-')[0] not in ids:

                            file_name = path + file #+ '_slice_' + str(slice)
                            if N ==20:
                                print(file_name)
                            im = cv2.imread(file_name)
                            im2 =cv2.resize( im[:,:,0] , (256,256))
                            im3 =cv2.resize( im[:,:,1] , (256,256))
                            im4 =cv2.resize( im[:,:,2] , (256,256))
                            im = np.stack((im2,im3,im4))
                            self.data.append(im)

                            if i ==0:
                                self.targets.append(0)
                            else:
                                self.targets.append(1)

                            c+=1

                            if c % 10000 == 0:
                                print(c)

                            self.original_files.append(file.split('_')[0] )


                            if i == 0:
                                self.targets_sev.append(0)
                            elif i ==1:
                               self.targets_sev.append(1)
                            elif i ==2:
                                self.targets_sev.append(2)
            for f in self.original_files:
                assert f in list(check['file'].loc[check['split'] =='validation'])
                assert f not in list(check['file'].loc[check['split'] =='train'])

            print('N is {}'.format(N))
            print('Lenght of data is {}'.format(len(self.data)))
            print('Slice balance is {}'.format(pd.DataFrame(self.targets).iloc[:,0].value_counts()))








    #    self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)

        label = target
        img = transforms.Normalize([-0.0859, -0.0859, -0.0859], [0.3467, 0.3467, 0.3467])(img)
        return img, label, int(self.targets_sev[index]), self.original_files[index]

    def __len__(self):
        return len(self.data)
