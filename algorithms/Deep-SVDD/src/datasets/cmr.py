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


class CMR_Dataset(data.Dataset):


    training_file = 'training.pt'
    test_file = 'test.pt'


    def __init__(self, root, pollution, N, normal_class, task, seed, data_split_path
    ) -> None:
        super().__init__()
        self.task = task  # training set or test set
        self.data_path = root
        self.normal_class = normal_class
        self.task = task

    #    classes = ['bottle' , 'cable',  'capsule',  'carpet',  'grid',  'hazelnut',  'leather',  'metal_nut',  'pill',  'screw',  'tile',  'toothbrush',  'transistor',  'wood' , 'zipper']
    #    folder = classes[normal_class]
        print('in the inti')
        scores = pd.read_csv(root+ '/IQA.csv')
        scores.columns = ['Image', 'score']
        normal_class =1
        normals = scores.loc[scores['score']==1]['Image'].reset_index(drop=True)
        random.seed(seed)
        idx= random.sample(list(range(0, len(normals))),N)
        train_files =normals[idx].reset_index(drop=True)

        ids = train_files.apply(lambda x: x.split('-')[0]).tolist()
        self.indexes = train_files.values.tolist()
        self.original_files = []


        self.data =[]
        self.targets=[]
        self.targets_sev = []

        #check its correct training data
        check = pd.read_csv(data_split_path + 'df_seed_' + str(seed) + '_n_' +str(N))

        for f in ids:
            assert f in list(check['file'].loc[check['split'] =='train'])




        if self.task == 'train':
            path = root +  '/ones/'

            for file in train_files:
                for slice in range(11):
                    file_name = path + file + '_slice_' + str(slice) +'.png'
                    im = cv2.imread(file_name)

                    try:
                        im2 =cv2.resize( im[:,:,0] , (64,64))
                        im3 =cv2.resize( im[:,:,1] , (64,64))
                        im4 =cv2.resize( im[:,:,2] , (64,64))
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
                            im = cv2.imread(file_name)
                            im2 =cv2.resize( im[:,:,0] , (64,64))
                            im3 =cv2.resize( im[:,:,1] , (64,64))
                            im4 =cv2.resize( im[:,:,2] , (64,64))
                            im = np.stack((im2,im3,im4))
                            self.data.append(im)

                            if i ==0:
                                self.targets.append(0)
                            else:
                                self.targets.append(1)

                            c+=1



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


    def __getitem__(self, index: int):



        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)

        label = target

        img = transforms.Normalize([-0.0859, -0.0859, -0.0859], [0.3467, 0.3467, 0.3467])(img)

        return img, label, index, int(self.targets_sev[index]), self.original_files[index]



    def __len__(self) -> int:
        return len(self.data)
