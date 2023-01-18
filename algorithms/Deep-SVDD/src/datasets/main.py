from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .mvtec import MVTEC_Dataset
from .fmnist import FMNIST_Dataset
from .mrart import MRART_Dataset
from .ixi import IXI_Dataset

def load_dataset(dataset_name, data_path, pollution, N, normal_class, task = None,seed=None,data_split_path=None):
    """Loads the dataset."""


    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, pollution = pollution, N=N,normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class)

    if dataset_name == 'mvtec':
        dataset = MVTEC_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class, task = task)

    if dataset_name == 'fmnist':
        dataset = FMNIST_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class)

    if dataset_name == 'mrart':
        dataset = MRART_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class,task=task,seed=seed, data_split_path = data_split_path)

    if dataset_name == 'ixi':
        dataset = IXI_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class,task=task,seed=seed, data_split_path = data_split_path)

    return dataset
