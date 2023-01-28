import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import random


class Dataset(data.Dataset):
    def __init__(self, root_dir, N, task, transform=None):
        super().__init__()
        self.paths = []
        self.labels=[]
        scores = pd.read_csv(root_dir+ '/derivatives/scores.tsv', sep='\t')



        normals = scores.loc[scores['score'] == 1].index.values
        random.seed(0)
        samp = random.sample((list(normals)), N)


        if task == 'train':
            self.indexes = samp
        else:
            self.indexes = [x for i,x in enumerate(scores.index.values) if (i not in samp) ]

        for im in self.indexes:
            self.paths.append(root_dir+ '/' + scores.iloc[im,0].split('_') [0] + '/anat/' + scores.iloc[im,0] + '.nii.gz')
            self.labels.append(scores.iloc[im,1])

        self.root_dir = root_dir
        self.task = task



        self.transform = transform






    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        indexes = list(range(0,len(self.paths)))



        file_path = self.paths[index]

        img = nib.load(file_path)
        array = np.array(img.dataobj)



     #   array = array.reshape(array.shape[2], array.shape[0], array.shape[1])


        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        if self.task == 'train':
            ind = np.random.randint(len(indexes) + 1) -1
            while (ind == index):
                ind = np.random.randint(len(indexes) + 1) -1

            file_path =  self.paths[indexes[ind]]

            img = nib.load(file_path)
            array2 = np.array(img.dataobj)

       #     array2 = array2.reshape(array2.shape[2], array2.shape[0], array2.shape[1])


            if self.transform:
                array2 = self.transform(array2)
            else:
                array2 = np.stack((array2,)*3, axis=1)
                array2 = torch.FloatTensor(array2)

            label = torch.FloatTensor([0])
            label2=label
        else:
            array2=array

            if self.labels[index] == 1:
                label = torch.FloatTensor([0])
            else:
                label = torch.FloatTensor([1])

            if (self.labels[index] == 1) | (self.labels[index] == 2):
                label2 = torch.FloatTensor([0])
            else:
                label2 = torch.FloatTensor([1])






        return array, array2, label, label2
