from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .mvtec import MVTEC_LeNet,MVTEC_LeNet_Autoencoder
from .fmnist_LeNet import FMNIST_LeNet,FMNIST_LeNet_Autoencoder


def build_network(net_name):
    """Builds the neural network."""


    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'MVTEC_LeNet':
        net = MVTEC_LeNet()

    if net_name == 'FMNIST_LeNet':
        net = FMNIST_LeNet()


    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""


    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'MVTEC_LeNet':
        ae_net = MVTEC_LeNet_Autoencoder()

    if net_name == 'FMNIST_LeNet':
        ae_net = FMNIST_LeNet_Autoencoder()

    return ae_net
