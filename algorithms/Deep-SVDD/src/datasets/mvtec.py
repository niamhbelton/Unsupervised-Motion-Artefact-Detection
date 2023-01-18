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

class MVTEC_Dataset(data.Dataset):


    training_file = 'training.pt'
    test_file = 'test.pt'


    def __init__(self, root, pollution, N, normal_class, task
    ) -> None:
        super().__init__()
        self.task = task  # training set or test set
        self.data_path = root
        self.normal_class = normal_class
        self.task = task

        classes = ['bottle' , 'cable',  'capsule',  'carpet',  'grid',  'hazelnut',  'leather',  'metal_nut',  'pill',  'screw',  'tile',  'toothbrush',  'transistor',  'wood' , 'zipper']
        folder = classes[normal_class]

        self.data =[]
        self.targets=[]

        if self.task == 'train':
            path2 = root + folder + '/train/good/'
            images = os.listdir(path2)
            for ind in range(len(images)):
                im = cv2.imread(path2 + images[ind])
                im2 =cv2.resize( im[:,:,0] , (64,64))
                im3 =cv2.resize( im[:,:,1] , (64,64))
                im4 =cv2.resize( im[:,:,2] , (64,64))
                im = np.stack((im2,im3,im4))
                self.data.append(im)

                self.targets.append(1)


        elif self.task == 'test':
            path2 = root + folder + '/test/'
            types = os.listdir(path2)
            for ty in types:
                path3 = path2 + ty
                images= os.listdir(path3)
                for image in images:
                    im = cv2.imread(path2 + ty + '/' + image)
                    im2 =cv2.resize( im[:,:,0] , (64,64))
                    im3 =cv2.resize( im[:,:,1] , (64,64))
                    im4 =cv2.resize( im[:,:,2] , (64,64))
                    im = np.stack((im2,im3,im4))
                    self.data.append(im)

                    if ty=='good':
                        self.targets.append(torch.Tensor([1]))
                    else:
                        self.targets.append(torch.Tensor([0]))

            print(len(self.data))
            print(len(self.targets))




    def __getitem__(self, index: int):



        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)

        label = target



        return img, label, index, None



    def __len__(self) -> int:
        return len(self.data)
