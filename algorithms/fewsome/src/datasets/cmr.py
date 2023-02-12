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
from torchvision import transforms

class CMR(data.Dataset):



    def __init__(self, indexes, root: str, normal_class,
            task, data_path, seed,N,sample,data_split_path):
        super().__init__()
        self.paths = []
        self.targets=[]
        self.targets_sev=[]
        self.indexes = indexes
        self.root_dir = root
        self.task = task
        scores = pd.read_csv(root+ '/IQA.csv')

        if indexes != []:
            check = pd.read_csv(data_split_path + 'df_seed_' + str(seed) + '_n_' +str(N))


        #normals = scores.loc[scores['score'] == 1].index.values
        #random.seed(0)
        #samp = random.sample((list(normals)), N)


        if self.indexes !=[]:
            if task =='train':
                for im in self.indexes:
                    assert scores.iloc[im,0].split('-') [0] in list(check['file'].loc[check['split'] =='train'])
                    self.paths.append(self.root_dir + 'data/' + scores.iloc[im,0].split('-')[0] + '-' + scores.iloc[im,0].split('-')[1] + '/' + scores.iloc[im,0] + '.nii.gz')



                    if scores.iloc[im,1] == 1:
                        targ = torch.FloatTensor([0])
                    else:
                        targ =torch.FloatTensor([1])
                    self.targets.append(targ)
                    self.targets_sev.append(targ)
            else:
                avoid = []
                names= []
                for im in self.indexes:
                    names.append( scores.iloc[im,0].split('-') [0])

                for i,s in enumerate(scores['Image'].apply(lambda x: x.split('-') [0])):
                    if s in names:
                        avoid.append(i)
                val_indexes =  [x for i,x in enumerate(scores.index.values) if (i not in self.indexes and i not in avoid) ]
                for im in val_indexes:
                    assert scores.iloc[im,0].split('-') [0] not in list(check['file'].loc[check['split'] =='train'])
                    assert (scores.iloc[im,0].split('_') [0] ) in list(check['file'].loc[check['split'] =='validation'])
                    self.paths.append(self.root_dir + 'data/' + scores.iloc[im,0].split('-')[0] + '-' + scores.iloc[im,0].split('-')[1] + '/' + scores.iloc[im,0] + '.nii.gz')
                    if scores.iloc[im,1] == 1:
                        self.targets.append(torch.FloatTensor([0]))
                    else:
                        self.targets.append(torch.FloatTensor([1]))

                    self.targets_sev.append(torch.FloatTensor([scores.iloc[im,1]-1]))

        else:
            for im in range(len(scores)):
                self.paths.append(self.root_dir + 'data/' + scores.iloc[im,0].split('-')[0] + '-' + scores.iloc[im,0].split('-')[1] + '/' + scores.iloc[im,0] + '.nii.gz')
                self.targets.append(scores.iloc[im,1])


    def __len__(self):
        return len(self.paths)



    def __getitem__(self, index: int, seed = 1, base_ind=-1):



        base=False
        target = int(self.targets[index])
        file_path = self.paths[index]

        img = nib.load(file_path)
        img = np.array(img.dataobj)
        img = torch.FloatTensor(img)


        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        img = torch.stack((img,img,img),1)
    #    for i in range(img.shape[0]):
    #        img[i] = transforms.Normalize([-0.0859, -0.0859, -0.0859], [0.3467, 0.3467, 0.3467])(img[i])



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

            target2 = int(self.targets[ind])
            file_path = self.paths[ind]

            img2 = nib.load(file_path)
            img2 = np.array(img2.dataobj)
            img2 = torch.FloatTensor(img2)
            img2 = img2.reshape(img2.shape[2], img2.shape[0], img2.shape[1])
            img2 = torch.stack((img2,img2,img2),1)
        #    for i in range(img2.shape[0]):
        #        img2[i] = transforms.Normalize([-0.0859, -0.0859, -0.0859], [0.3467, 0.3467, 0.3467])(img2[i])




            label = torch.FloatTensor([0])
        else:
            img2 = torch.Tensor([1])
            label = target


#        print(img.shape)
        return img, img2, label, base, int(self.targets_sev[index])
