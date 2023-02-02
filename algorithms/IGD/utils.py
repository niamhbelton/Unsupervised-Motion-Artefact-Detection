import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from p256.mrart_data_loader import MRART
# import faiss
from p256.ixi_data_loader import IXI

# import ResNet


transform_color = transforms.Compose([transforms.Resize(256),
                                      # transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 # transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    # index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32
    np.random.seed(15) # worker_seed
    random.seed(15) # worker_seed

def get_loaders(num_images, dataset, label_class, batch_size, seed, data_path,data_split_path):
    if dataset in ['cifar10', 'mnist', 'fashion', 'mrart','ixi']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10  #uncomment on new run
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse) #downld True
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse) #downld True
        elif dataset == "mnist":
            ds = torchvision.datasets.MNIST  #uncomment on new run
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse) #downld True
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse) #downld True
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_mrart
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        elif (dataset == "mrart") :
            coarse = {}
            trainset = MRART(data_path, label_class,
                    'train', seed, num_images, None, data_split_path)
            testset = MRART(data_path, label_class,
                    'test', seed, num_images, None, data_split_path)

        elif dataset == 'ixi':
            coarse = {}
            trainset = IXI(data_path, label_class,
                    'train', seed, num_images, None, data_split_path)
            testset = IXI(data_path, label_class,
                    'test', seed, num_images, None,data_split_path)



        if (dataset != 'mrart') & (dataset != 'ixi'):
            idx = np.where(np.array(trainset.targets) == label_class)[0]
            trainset.data = trainset.data[idx]
            trainset.targets = [0]*len(trainset.data)
            trainset.data = trainset.data[random_ids,:,:]  #[num_images: (2*num_images)] #[270:300]

            #for i,ind in enumerate(range(len(random_ids))):
            #    if i ==0:
            #        ids = np.arange(random_ids[ind]*192, (random_ids[ind] * 192)+192).tolist()
            #    else:
            #        ids = ids + np.arange(random_ids[ind]*192, (random_ids[ind] * 192)+192).tolist()




            trainset.targets = [0]*len(trainset.data)#.targets[num_images: (2*num_images)] #[270:300]



            for i in range(len(testset.targets)):
                if testset.targets[i] == label_class:
                    testset.targets[i] = 0
                else:
                    testset.targets[i] = 1

        print('\ntrainset.data shape:', len(trainset.data))


        g = torch.Generator()
        g.manual_seed(0)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False) # False # , worker_init_fn=seed_worker, generator=g
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False) #False # , worker_init_fn=seed_worker, generator=g
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()


def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)
