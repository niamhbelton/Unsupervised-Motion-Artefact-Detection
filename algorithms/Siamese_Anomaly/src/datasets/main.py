from .mnist import MNIST
from .cifar10 import CIFAR10
from .mnist_fashion import FASHION
from .mvtec import MVTEC
from .mrart import MRART
from .mrart_vote import MRART_vote
from .ixi import ixi


def load_dataset(dataset_name, indexes, normal_class, task, data_path, download_data,seed=None, N=None,sample=None,data_split_path=None,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'fashion','mvtec', 'mrart')

    dataset = None


    if dataset_name == 'mnist':
        dataset = MNIST(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                download_data = download_data)





    if dataset_name == 'cifar10':
        dataset = CIFAR10(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                download_data = download_data)

    if dataset_name == 'fashion':
        dataset = FASHION(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                download_data = download_data)


    if dataset_name == 'mvtec':
        dataset = MVTEC(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                seed=seed,
                                N=N)
    if dataset_name == 'mrart':
        print(seed)
        print(N)
        print(data_split_path)
        dataset = MRART(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                seed=seed,
                                N=N,
                                sample=sample,
                                data_split_path=data_split_path)

    if dataset_name == 'mrart_vote':
        dataset = MRART_vote(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                seed=seed,
                                N=N,
                                sample=sample,
                                data_split_path=data_split_path)

    if dataset_name == 'ixi':
        dataset = ixi(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                seed=seed,
                                N=N,

                                sample=sample,
                                data_split_path=data_split_path)


    return dataset
