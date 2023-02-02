import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
# import faiss
# import ResNet

torch.manual_seed(torch.seed()) #(0)
# np.random.seed(15)


mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

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

# def get_resnet_model(resnet_type=152):
#     """
#     A function that returns the required pre-trained resnet model
#     :param resnet_number: the resnet type
#     :return: the pre-trained model
#     """
#     if resnet_type == 18:
#         return ResNet.resnet18(pretrained=True, progress=True)
#     elif resnet_type == 50:
#         return ResNet.wide_resnet50_2(pretrained=True, progress=True)
#     elif resnet_type == 101:
#         return ResNet.resnet101(pretrained=True, progress=True)
#     else:  #152
#         return ResNet.resnet152(pretrained=True, progress=True)


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
    
def get_loaders(num_images, dataset, label_class, batch_size):
    if dataset in ['cifar10', 'mnist', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10  #uncomment on new run
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=False, transform=transform, **coarse) #downld True
            testset = ds(root='data', train=False, download=False, transform=transform, **coarse) #downld True
        elif dataset == "mnist":
            ds = torchvision.datasets.MNIST  #uncomment on new run
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse) #downld True
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse) #downld True
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        
        
        testset.targets = [int(t != label_class) for t in testset.targets]
        
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        
        # print('\ntrainset.data shape:', len(trainset.data))
        # print('trainset.data [:30] shape:', len(trainset.data[:30]), '\n')
        trainset.data = trainset.data[num_images: (2*num_images)] #[270:300]
        
        # print('\ntrainset.targets shape:', len(trainset.targets))
        # print('trainset.targets[:30] shape:', len(trainset.targets[:30]), '\n')
        trainset.targets = trainset.targets[num_images: (2*num_images)] #[270:300]
        
        g = torch.Generator()
        g.manual_seed(0)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False) # False # , worker_init_fn=seed_worker, generator=g
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False) #False # , worker_init_fn=seed_worker, generator=g
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


