import torch

import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pandas as pd
import torchvision
from torch import autograd
from torch import optim
import torch.nn.init as init
from timeit import default_timer as timer
import time
import cv2
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm
import numpy as np
# from ... import Helper
# from ... import Recorder
import Helper
from Recorder import Recorder
from p256.ssim_module import *
from p256.mvtec_module import twoin1Generator256, VisualDiscriminator256, Encoder_256
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from p256.mvtex_data_loader import *

from torch.autograd import Variable
import argparse

from p256.cifar_data_loader import *
import utils

# numpy.random.seed(15)
# np.random.seed(15)

device = torch.device("cuda:1")
# print(">> Device Info: {} is in use".format(device))
parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('-n', '--num', nargs='+', type=int, help='<Required> Set flag', required=True)
parser.add_argument('-sr', '--sample_rate', default=1, type=float)
parser.add_argument( '--dataset', default='mnist', type=str)
parser.add_argument( '--exp_name',type=str)
parser.add_argument( '--data_path',type=str, help='Required for MR-ART and IXI dataset')
parser.add_argument( '--data_split_path',type=str, help='Required for IXI dataset')


DIM = 32  # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 3  # Number of GPUs
# BATCH_SIZE = 15 # so it's half of train size, which is 30 images to comply with Belton's train size.  # Batch size. Must be a multiple of N_GPUS

BATCH_SIZE = 64

LAMBDA = 10  # Gradient pena1lty lambda hyperparameter

MAX_EPOCH = 256 #128 #256 #2 #10 #
############################ Parameters ############################
latent_dimension =  128 #

msssim_weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

category = {
    1: "zero",
    2: "one",
    3: "two",
    4: "three",
    5: "four",
    6: "five",
    7: "six",
    8: "seven",
    9: "eight",
    10: "nine"
    # ,
    # 11: "cable",
    # 12: "transistor",
    # 13: "toothbrush",
    # 14: "screw",
    # 15: "zipper"
}

# category = {"pill"}
data_range = 2.1179 + 2.6400
####################################################################

USE_SSIM = True

LR = 1e-4  # 0.0001


mse_criterion = torch.nn.MSELoss()
l1_criterion = torch.nn.L1Loss()
bce_criterion = torch.nn.BCELoss()

sigbce_criterion = torch.nn.BCEWithLogitsLoss()

auc_max_store = []
f1_store=[]
acc_store=[]

def create_dir(dir):
    if not os.path.exists(dir):
            os.mkdir(path=dir)

def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def load_train(sample_rate, label_class, num_images, BATCH_SIZE, seed):
    print('the label class is {}'.format(label_class))
    train_data_loader, _ = utils.get_loaders(num_images, dataset=args.dataset, label_class=label_class, batch_size=BATCH_SIZE,seed=seed, data_path =args.data_path)
    return train_data_loader, num_images #imagenet_data.__len__()

def load_test(label_class, BATCH_SIZE, sample_rate,seed):

    _, valid_data_loader = utils.get_loaders(num_images, dataset=args.dataset, label_class=label_class, batch_size=BATCH_SIZE, seed=seed, data_path =args.data_path)
    return valid_data_loader


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if max_iter == 0:
        raise Exception("ERROR")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def init_c(DataLoader, net, eps=0.1):
    generator.c = None
    c = torch.zeros((1, latent_dimension)).to(device)
    net.eval()
    n_samples = 0
    with torch.no_grad():
        for index, (images, label,_,_) in enumerate(DataLoader):

            # get the inputs of the batch
            img = images.to(device)

            outputs = net.encoder(img)

            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
            del img
            del images
            del outputs
            torch.cuda.empty_cache()


    c /= n_samples

    del n_samples
    torch.cuda.empty_cache()

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


sig_f = 1
def init_sigma(DataLoader, net):
    generator.sigma = None
    net.eval()
    tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(device)
    n_samples = 0
    with torch.no_grad():
        for index, (images, label,_,_) in enumerate(DataLoader):
            img = images.to(device)

            latent_z = net.encoder(img).detach()
            diff = (latent_z - net.c )** 2
            tmp = torch.sum(diff.detach(), dim=1)
            if (tmp.mean().detach() / sig_f) < 1:
                tmp_sigma += 1
            else:
                tmp_sigma += tmp.mean().detach() / sig_f
            n_samples += 1
            del img
            del images

            del tmp
            torch.cuda.empty_cache()


    tmp_sigma /= n_samples
    net.c.to(device)
    return tmp_sigma


def train(args, NORMAL_NUM,
          generator, discriminator,
          optimizer_g, optimizer_d, train_size, BATCH_SIZE, seed,dataset):



    AUC_LIST = []
    F1_LIST=[]
    ACC_LIST=[]
    AUC_LIST2 = []
    F1_LIST2=[]
    ACC_LIST2=[]
    SEV_MAX_LIST=[]
    SEV_MEAN_LIST=[]
    global test_auc
    test_auc = 0
    generator.c = None
    generator.sigma = None

    START_ITER = 0

    BEST_AUC = 0
    generator.train()
    discriminator.train()
    train_dataset_loader, valid_data_loader = utils.get_loaders(num_images, dataset=args.dataset, label_class=NORMAL_NUM, batch_size=BATCH_SIZE, seed=seed, data_path =args.data_path, data_split_path = args.data_split_path)

    #train_dataset_loader, _ = load_train(args.sample_rate, NORMAL_NUM, train_size, BATCH_SIZE, seed)


    END_ITER = int((train_size / BATCH_SIZE) * MAX_EPOCH)
    torch.cuda.empty_cache()


    generator.c = init_c(train_dataset_loader, generator)
    torch.cuda.empty_cache()


    generator.c.requires_grad = False

    generator.sigma = init_sigma(train_dataset_loader, generator)
    torch.cuda.empty_cache()


    generator.sigma.requires_grad = False

    # generator.sigma =1.0749 # static value copied from original mvtec code, coz its too big

    train_data = iter(train_dataset_loader)
    process = tqdm(range(START_ITER, END_ITER), desc='{AUC: }')
    # print('process: ', process, START_ITER, END_ITER)
    training_time = 0
    tts=[]
    inf_times=[]
    best_test_auc = 0
    max_patience =5
    patience=0


    for iteration in process:
        start_time = time.perf_counter()
        poly_lr_scheduler(optimizer_d, init_lr=LR, iter=iteration, max_iter=END_ITER)
        poly_lr_scheduler(optimizer_g, init_lr=LR, iter=iteration, max_iter=END_ITER)

        # --------------------- Loader ------------------------
        batch = next(train_data, None)
        if batch is None:
            train_data = iter(train_dataset_loader)
            batch = next(train_data)#.next()
        # print('batch: ', len(batch), 'train_data: ', len(train_data))
        batch = batch[0]  # batch[1] contains labels
        real_data = batch.to(device)
        torch.cuda.empty_cache()


    #    if args.dataset == 'mrart':
    #        real_data = real_data.reshape((real_data.shape[0]*real_data.shape[1], real_data.shape[2], real_data.shape[3], real_data.shape[4]))


        # --------------------- TRAIN E ------------------------
        optimizer_g.zero_grad()
        b, c, _, _ = real_data.shape
        latent_z = generator .encoder(real_data)

        torch.cuda.empty_cache()


        fake_data = generator .generate(latent_z)


        torch.cuda.empty_cache()

    #    print(torch.cuda.memory_summary('cuda:2'))





        # Reconstruction loss
        weight = 0.85
        ms_ssim_batch_wise = 1 - ms_ssim(real_data, fake_data, data_range=data_range,
                                         size_average=True, win_size=11, weights=msssim_weight)
        l1_batch_wise = l1_criterion(real_data, fake_data) / data_range
        ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

        ############ Interplote ############
        e1 = torch.flip(latent_z, dims=[0])
        alpha = torch.FloatTensor(b, 1).uniform_(0, 0.5).to(device)
        e2 = alpha * latent_z + (1 - alpha) * e1
        g2 = generator .generate(e2)
        reg_inter = torch.mean(discriminator (g2) ** 2)

        ############ GAC ############
        diff = (latent_z - generator .c) ** 2
        dist = -1 * (torch.sum(diff, dim=1) / generator .sigma)
        svdd_loss = torch.mean(1 - torch.exp(dist))

        encoder_loss = ms_ssim_l1 + svdd_loss + 0.1 * reg_inter
        encoder_loss.backward()
        optimizer_g.step()

        ############ Discriminator ############
        optimizer_d.zero_grad()
        g2 = generator .generate(e2).detach()
        fake_data = generator (real_data).detach()
        d_loss_front = torch.mean((discriminator (g2) - alpha) ** 2)
        gamma = 0.2
        tmp = fake_data + gamma * (real_data - fake_data)
        d_loss_back = torch.mean(discriminator (tmp) ** 2)
        d_loss = d_loss_front + d_loss_back
        d_loss.backward()
        optimizer_d.step()

        if iteration != 0 :
            if iteration % int((train_size / BATCH_SIZE) * 10) == 0 and iteration != 0:
                generator .sigma = init_sigma(train_dataset_loader, generator)
                generator .c = init_c(train_dataset_loader, generator)

        end_time = time.perf_counter()
        training_time += end_time - start_time
        tts.append(training_time)
        # ------------------ RECORDER ------------------
#         if recorder is not None:
#             recorder.record(loss=svdd_loss, epoch=int(iteration / BATCH_SIZE),
#                             num_batches=len(train_data), n_batch=iteration, loss_name='GAC')

#             recorder.record(loss=torch.mean(dist), epoch=int(iteration / BATCH_SIZE),
#                             num_batches=len(train_data), n_batch=iteration, loss_name='DIST')

#             recorder.record(loss=ms_ssim_batch_wise, epoch=int(iteration / BATCH_SIZE),
#                             num_batches=len(train_data), n_batch=iteration, loss_name='MS-SSIM')

#             recorder.record(loss=l1_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            # num_batches=len(train_data), n_batch=iteration, loss_name='L1')

        # print('iter: ', iteration, 'train size: ', (train_size))
        print('Training time at iteration {} is {}, loss is {}'.format(iteration, training_time, d_loss ))
    #    if (iteration % int((train_size / BATCH_SIZE) * 20) == 0 and iteration > 0)  or iteration == END_ITER - 1: # * 5
        is_end = True if iteration == END_ITER - 1 else False
        if (dataset == 'mrart') | (dataset == 'ixi'):
            test_auc, AUC_LIST, F1_LIST, ACC_LIST, inf_time, auc_result2, AUC_LIST2, F1_LIST2, ACC_LIST2,SEV_MAX_LIST, SEV_MEAN_LIST,df= validation_vote(valid_data_loader, NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, F1_LIST, ACC_LIST, AUC_LIST2, F1_LIST2, ACC_LIST2 , SEV_MAX_LIST, SEV_MEAN_LIST , END_ITER, BATCH_SIZE,dataset)

        else:
            test_auc, AUC_LIST, F1_LIST, ACC_LIST, inf_time = validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, F1_LIST, ACC_LIST, AUC_LIST2, F1_LIST2, ACC_LIST2, END_ITER, BATCH_SIZE,dataset)

        inf_times.append(inf_time)
        process.set_description("{AUC: %.5f}" % test_auc)
        print('AUC score is {}'.format(AUC_LIST[-1]))
        print('acc score is {}'.format(ACC_LIST[-1]))
        print('f1 score is {}'.format(F1_LIST[-1]))
        print('inference time: {}'.format( inf_time))

        if (dataset == 'mrart' )| (dataset == 'ixi'):
            print('Based on Mean:')
            process.set_description("{AUC: %.5f}" % auc_result2)
            print('AUC score is {}'.format(AUC_LIST2[-1]))
            print('acc score is {}'.format(ACC_LIST2[-1]))
            print('f1 score is {}'.format(F1_LIST2[-1]))
            print('inference time: {}'.format( inf_time))
            print('Acc sev based on mean is  {}'.format(SEV_MEAN_LIST[-1]))
            print('Acc sev based on max is  {}'.format(SEV_MAX_LIST[-1]))



        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_df = df
            patience=0
        else:
            patience+=1

        if patience == max_patience:
            break



        #    process.set_description("{F1: %.5f}" % F1_LIST[-1])
        #    process.set_description("{ACC: %.5f}" % ACC_LIST[-1])
            # opt_path = ckpt_path + '/optimizer'
            # if not os.path.exists(opt_path):
            #     os.mkdir(path=opt_path)
            # torch.save(optimizer_g.state_dict(), ckpt_path + '/optimizer/g_opt.pth')
            # torch.save(optimizer_d.state_dict(), ckpt_path + '/optimizer/d_opt.pth')
    print('---------------------\n auc max: ', max(AUC_LIST), ', ')
    max_ind = np.argmax(AUC_LIST)
    print('---------------------\n on epoch: ', max_ind, ', ')
    print('---------------------\n F1: ', F1_LIST[max_ind], ', ')
    print('---------------------\n ACC: ', ACC_LIST[max_ind], ', ')
    print('---------------------\n Training Time: ', tts[max_ind], ', ')
    print('---------------------\n Inference Time: ', inf_times[max_ind], ', ')

    auc_max_store.append(max(AUC_LIST))
    f1_store.append(F1_LIST[max_ind])
    acc_store.append(ACC_LIST[max_ind])


    file_name = 'results_seed_' + str(seed) + '_train_size_' + str(train_size)+ '_maxind_' + str(max_ind)
    best_df.to_csv(file_name)

    print('time: ', training_time, '\n------------------------\n')

    print('Check inference time...')
    if (dataset == 'mrart') | (dataset == 'ixi'):
        one_pass_inf = inf(generator, discriminator,dataset,train_size, NORMAL_NUM,seed,args.data_path)

    print('Inference time for one pass is {}'.format(one_pass_inf))

def validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, F1_LIST, ACC_LIST, END_ITER, BATCH_SIZE,dataset):
    discriminator.eval()
    generator.eval()
    # resnet.eval()
    y = []
    score = []
    normal_gsvdd = []
    abnormal_gsvdd = []
    normal_recon = []
    abnormal_recon = []
    # test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)

    with torch.no_grad():
        for i in range(1): # range(len(list_test)): #
            start_inf = time.perf_counter()
            valid_dataset_loader = load_test(NORMAL_NUM, BATCH_SIZE, sample_rate=1., seed=seed)

            # print('valid_dataset_loader: ', len(valid_dataset_loader))
            for index, (images, label) in enumerate(valid_dataset_loader):
                # print('label: ', label)
                # print('ims: ', len(images), 'labels: ', len(label))
                img = images.to(device)
                # latent_z = generator.encoder(img)
                latent_z = generator .encoder(img)
                generate_result = generator (img)

                ############################## Normal #####################

                for visual_index in range(latent_z.shape[0]):
                    weight = 0.85
                    tmp_org_unsq = img[visual_index].unsqueeze(0)
                    tmp_rec_unsq = generate_result[visual_index].unsqueeze(0)

                    ms_ssim_batch_wise = 1 - ms_ssim(tmp_org_unsq, tmp_rec_unsq, data_range=data_range,
                                                     size_average=True, win_size=11, weights=msssim_weight)
                    l1_loss = l1_criterion(img[visual_index], generate_result[visual_index]) / data_range
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_loss

                    diff = (latent_z[visual_index] - generator.c) ** 2
                    dist = -1 * torch.sum(diff, dim=1) / generator.sigma
                    guass_svdd_loss = 1 - torch.exp(dist)

                    anormaly_score = (0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).cpu().detach().numpy()
                    score.append(float(anormaly_score))

                    la = label[visual_index]  # .cpu().detach().numpy()

                    if la == 0: #"good":
                        y.append(0)
                    else:
                        y.append(1)
            ###################################################
    # print('y: ', y, 'score: ', score)
    # print('sum of labels: ', s)
    end_inf =time.perf_counter()
    fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=1)
    outputs = np.array(score)
    thres = np.percentile(outputs, 10)
    outputs[outputs > thres] =1.0
    outputs[outputs <= thres] =0.0
    f1 = f1_score(np.array(y),outputs)


    df = pd.concat([pd.DataFrame(y), pd.DataFrame(outputs)], axis=1).reset_index(drop=True)
    df.columns=['y','outputs']

    fp = len(df.loc[(df['outputs']== 1 ) & (df['y'] == 0)])
    tn = len(df.loc[(df['outputs']== 0) & (df['y'] == 0)])
    fn = len(df.loc[(df['outputs'] == 0) & (df['y'] == 1)])
    tp = len(df.loc[(df['outputs'] == 1) & (df['y'] == 1)])
    spec = tn / (fp + tn)
    recall = tp / (tp+fn)
    acc = (recall + spec) / 2

    # print('fpr: ', fpr, 'tpr: ', tpr)
    auc_result = auc(fpr, tpr)
    AUC_LIST.append(auc_result)
    F1_LIST.append(f1)
    ACC_LIST.append(acc)
    inf_time = end_inf - start_inf



    # auc_result = roc_auc_score(y, score)
    # AUC_LIST.append(auc_result)
    # tqdm.write(str(auc_result), end='.....................')
    # auc_file = open(ckpt_path + "/auc.txt", "a")
    # auc_file.write('Iter {}:            {}\r\n'.format(str(iteration), str(auc_result)))
    # auc_file.close()
    # if iteration == END_ITER - 1:
    #     auc_file = open(ckpt_path + "/auc.txt", "a")
    #     auc_file.write('BEST AUC -> {}\r\n'.format(max(AUC_LIST)))
    #     auc_file.close()

    return auc_result, AUC_LIST, F1_LIST, ACC_LIST, inf_time

def validation_vote(valid_dataset_loader, NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, F1_LIST, ACC_LIST, AUC_LIST2, F1_LIST2, ACC_LIST2 , SEV_MEAN_LIST, SEV_MAX_LIST, END_ITER, BATCH_SIZE,dataset):
    discriminator.eval()
    generator.eval()
    # resnet.eval()
    y = []
    y_sev = []
    score = []
    score2=[]
    normal_gsvdd = []
    abnormal_gsvdd = []
    normal_recon = []
    abnormal_recon = []
    # test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    dict_score1={}
    dict_label={}
    dict_label_sev={}
    used = []
    orig_files=[]

    with torch.no_grad():
        for i in range(1): # range(len(list_test)): #
            start_inf = time.perf_counter()
        #    valid_dataset_loader = load_test(NORMAL_NUM, 1, sample_rate=1., seed=seed)

            for index, (images, label, label_sev, original_file) in enumerate(valid_dataset_loader):

                # print('label: ', label)
                # print('ims: ', len(images), 'labels: ', len(label))
                img = images.to(device)
                # latent_z = generator.encoder(img)
                latent_z = generator .encoder(img)
                generate_result = generator (img)

                ############################## Normal #####################

                for visual_index in range(latent_z.shape[0]):
                    weight = 0.85
                    tmp_org_unsq = img[visual_index].unsqueeze(0)
                    tmp_rec_unsq = generate_result[visual_index].unsqueeze(0)

                    ms_ssim_batch_wise = 1 - ms_ssim(tmp_org_unsq, tmp_rec_unsq, data_range=data_range,
                                                     size_average=True, win_size=11, weights=msssim_weight)
                    l1_loss = l1_criterion(img[visual_index], generate_result[visual_index]) / data_range
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_loss

                    diff = (latent_z[visual_index] - generator .c) ** 2
                    dist = -1 * torch.sum(diff, dim=1) / generator .sigma
                    guass_svdd_loss = 1 - torch.exp(dist)

                    anomaly_score = (0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).cpu().detach().numpy()
                    if original_file[visual_index] in used:
                        dict_score1['{}'.format(original_file[visual_index])].append(anomaly_score)#.cpu().data.numpy())
                        dict_label['{}'.format(original_file[visual_index])]=label[visual_index].cpu().data.numpy().tolist()
                        dict_label_sev['{}'.format(original_file[visual_index])]=label_sev[visual_index].cpu().data.numpy().tolist()
                        orig_files.append(original_file[visual_index])
                    else:
                        dict_score1['{}'.format(original_file[visual_index])] = []
                        dict_score1['{}'.format(original_file[visual_index])].append(anomaly_score)#.cpu().data.numpy())
                        dict_label['{}'.format(original_file[visual_index])]=label[visual_index].cpu().data.numpy().tolist()
                        dict_label_sev['{}'.format(original_file[visual_index])] = label_sev[visual_index].cpu().data.numpy().tolist()
                        orig_files.append(original_file[visual_index])
                        used.append(original_file[visual_index])


                    score.append(float(anomaly_score))

                    la = label[visual_index]  # .cpu().detach().numpy()

                    if la == 0: #"good":
                        y.append(0)
                    else:
                        y.append(1)

                    if int((index * BATCH_SIZE) + visual_index) % 10000 ==0:
                        print(int((index * BATCH_SIZE) + visual_index))






    end_inf =time.perf_counter()


    score1 = pd.DataFrame.from_dict(dict_score1, orient='index').T
    score2 = pd.DataFrame.from_dict(dict_score1, orient='index').T
    s1=pd.DataFrame(score1.mean()).reset_index()
    s1.columns = ['file', 's1']
    s2 = pd.DataFrame(score2.max()).reset_index()
    s2.columns = ['file', 's2']
    labs = pd.DataFrame([dict_label]).T.reset_index()
    labs.columns = ['file', 'label']
    labs_sev = pd.DataFrame([dict_label_sev]).T.reset_index()
    labs_sev.columns = ['file', 'label_sev']
    df=pd.merge(s1, s2,  on='file')
    df=pd.merge(df, labs, on ='file')
    df2=pd.merge(df, labs_sev, on ='file')

    labels = np.array(df2['label'])
    label_sev = np.array(df2['label_sev'])
    scores1 = np.array(df2['s1'])
    scores2 = np.array(df2['s2'])

    print(df2['label'].value_counts())
    #assert df2['label'].value_counts().values[0] == df2['label'].value_counts().values[1]

    print(df2)
    print(labels)
    print(scores1)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores1)
    perc_thres = int((len(df2['label'].loc[df2['label'] ==0]) / (len(df2['label'].loc[df2['label'] ==1]) + len(df2['label'].loc[df2['label'] ==0])))*100)
    thresh = np.percentile(scores1, perc_thres)
    y_pred = np.where(scores1 >= thresh, 1, 0)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="binary")
    fp = len(df2.loc[(y_pred== 1 ) & (df2['label'] == 0)])
    tn = len(df2.loc[(y_pred== 0) & (df2['label'] == 0)])
    fn = len(df2.loc[(y_pred == 0) & (df2['label'] == 1)])
    tp = len(df2.loc[(y_pred== 1) & (df2['label'] == 1)])
    spec = tn / (fp + tn)
    recall = tp / (tp+fn)
    acc = (recall + spec) / 2

    auc_result = auc(fpr, tpr)
    AUC_LIST.append(auc_result)
    F1_LIST.append(f1)
    ACC_LIST.append(acc)
    inf_time = end_inf - start_inf




    #score 2
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores2)
    thresh = np.percentile(scores2, perc_thres)
    y_pred = np.where(scores2 >= thresh, 1, 0)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="binary")
    fp = len(df2.loc[(y_pred== 1 ) & (df2['label'] == 0)])
    tn = len(df2.loc[(y_pred== 0) & (df2['label'] == 0)])
    fn = len(df2.loc[(y_pred == 0) & (df2['label'] == 1)])
    tp = len(df2.loc[(y_pred== 1) & (df2['label'] == 1)])
    spec = tn / (fp + tn)
    recall = tp / (tp+fn)
    acc = (recall + spec) / 2
    auc_result2 = auc(fpr, tpr)
    AUC_LIST2.append(auc_result2)
    F1_LIST2.append(f1)
    ACC_LIST2.append(acc)


    #severity accuracy
    perc_thres2 = int(((len(df2['label_sev'].loc[df2['label_sev'] ==0]) + len(df2['label_sev'].loc[df2['label_sev'] ==1])) / (len(df2['label_sev'].loc[df2['label_sev'] ==2])  + len(df2['label_sev'].loc[df2['label_sev'] ==1]) + len(df2['label_sev'].loc[df2['label_sev'] ==0])))*100)
    thres = np.percentile(scores1, perc_thres)
    thres2 = np.percentile(scores1, perc_thres2)
    outputs= scores1.copy()
    outputs[outputs > thres2] =2
    outputs[(outputs <= thres2) & (outputs > thres)] =1
    outputs[(outputs <= thres)] =0
    cl0 = len(outputs[(outputs == 0 ) & (label_sev == 0)])
    cl1 = len(outputs[(outputs == 1 ) & (label_sev == 1)])
    cl2 = len(outputs[(outputs == 2 ) & (label_sev == 2)])
    accuracy_sev1 = (cl0+cl1+cl2) / len(scores1)
    print('Accuracy with severity {}'.format(accuracy_sev1))


    thres = np.percentile(scores2, perc_thres)
    thres2 = np.percentile(scores2, perc_thres2)
    outputs= scores2.copy()
    outputs[outputs > thres2] =2
    outputs[(outputs <= thres2) & (outputs > thres)] =1
    outputs[(outputs <= thres)] =0
    cl0 = len(outputs[(outputs == 0 ) & (label_sev == 0)])
    cl1 = len(outputs[(outputs == 1 ) & (label_sev == 1)])
    cl2 = len(outputs[(outputs == 2 ) & (label_sev == 2)])
    accuracy_sev2 = (cl0+cl1+cl2) / len(scores1)
    print('Accuracy with severity {}'.format(accuracy_sev2))

    SEV_MAX_LIST.append(accuracy_sev2)
    SEV_MEAN_LIST.append(accuracy_sev1)


    # auc_result = roc_auc_score(y, score)
    # AUC_LIST.append(auc_result)
    # tqdm.write(str(auc_result), end='.....................')
    # auc_file = open(ckpt_path + "/auc.txt", "a")
    # auc_file.write('Iter {}:            {}\r\n'.format(str(iteration), str(auc_result)))
    # auc_file.close()
    # if iteration == END_ITER - 1:
    #     auc_file = open(ckpt_path + "/auc.txt", "a")
    #     auc_file.write('BEST AUC -> {}\r\n'.format(max(AUC_LIST)))
    #     auc_file.close()

    return auc_result, AUC_LIST, F1_LIST, ACC_LIST, inf_time, auc_result2, AUC_LIST2, F1_LIST2, ACC_LIST2, SEV_MEAN_LIST, SEV_MAX_LIST,df2



def inf(generator, discriminator,dataset,num_images, NORMAL_NUM,seed,data_path):
    discriminator.eval()
    generator.eval()
    # resnet.eval()
    y = []
    y_sev = []
    score = []
    score2=[]
    normal_gsvdd = []
    abnormal_gsvdd = []
    normal_recon = []
    abnormal_recon = []
    train_dataset_loader, valid_dataset_loader = utils.get_loaders(num_images, dataset=dataset, label_class=NORMAL_NUM, batch_size=1, seed=seed, data_path =args.data_path, data_split_path = args.data_split_path)


    with torch.no_grad():
        for i in range(1): # range(len(list_test)): #
            start_inf = time.perf_counter()
        #    valid_dataset_loader = load_test(NORMAL_NUM, 1, sample_rate=1., seed=seed)

            for index, (images, label, label_sev, original_file) in enumerate(valid_dataset_loader):

                # print('label: ', label)
                # print('ims: ', len(images), 'labels: ', len(label))
                img = images.to(device)
                # latent_z = generator.encoder(img)
                latent_z = generator .encoder(img)
                generate_result = generator (img)

                ############################## Normal #####################

                for visual_index in range(latent_z.shape[0]):
                    weight = 0.85
                    tmp_org_unsq = img[visual_index].unsqueeze(0)
                    tmp_rec_unsq = generate_result[visual_index].unsqueeze(0)

                    ms_ssim_batch_wise = 1 - ms_ssim(tmp_org_unsq, tmp_rec_unsq, data_range=data_range,
                                                     size_average=True, win_size=11, weights=msssim_weight)
                    l1_loss = l1_criterion(img[visual_index], generate_result[visual_index]) / data_range
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_loss

                    diff = (latent_z[visual_index] - generator .c) ** 2
                    dist = -1 * torch.sum(diff, dim=1) / generator .sigma
                    guass_svdd_loss = 1 - torch.exp(dist)

                    anomaly_score = (0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).cpu().detach().numpy()

                    end_inf =time.perf_counter()
                    break
                break



    return end_inf - start_inf


if __name__ == "__main__":




    args = parser.parse_args()
    NORMAL_NUM_LIST = set(args.num)
    # sample_rate = args.sample_rate


    train_size_choices = [ 10,20,30]

    seeds = [1001, 985772, 128037, 875688, 71530]

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for num_images in train_size_choices:
            if(num_images <= 50):
                BATCH_SIZE = num_images
            elif (num_images == 100):
                BATCH_SIZE =50

            if (args.dataset == 'mrart'  )| (args.dataset == 'ixi'):
                BATCH_SIZE = 24



            print('---------------------------------------------------')
            print(str(num_images), ' = size of training dataset. START.')
            print('---------------------------------------------------')

            for i in args.num:

                NORMAL_NUM = i -1
                # NORMAL_NUM = category[key] if key.isdigit() else key
                print('Current Item: {}'.format(NORMAL_NUM))



                Experiment_name = args.exp_name + '_{}'.format(str(NORMAL_NUM))

                # recorder = Recorder(Experiment_name, 'MVTec_No.{}'.format(str(NORMAL_NUM)))

                save_root_path = './p256/check_points'
        #         create_dir(save_root_path)
        #         save_root_path = os.path.join(save_root_path, "IGD_wo_inter")
        #         create_dir(save_root_path)
        #         ckpt_path = os.path.join(save_root_path, "p256_SR-{}".format(args.sample_rate))
        #         create_dir(ckpt_path)
        #         ckpt_path = os.path.join(ckpt_path, Experiment_name)

        #         if not os.path.exists(ckpt_path):
        #             os.mkdir(path=ckpt_path)
        #         auc_file = open(ckpt_path + "/auc.txt", "w")
        #         auc_file.close()

                generator = twoin1Generator256(64, latent_dimension=latent_dimension) #64
                discriminator = VisualDiscriminator256(64) #64
                generator.to(device)
                discriminator.to(device)



                for param in generator.pretrain.parameters():
                    param.requires_grad = False




                optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0, 0.9), weight_decay=1e-6)
                optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0, 0.9))


                print('seed {}'.format(seed))
                print('num_images {}'.format(num_images))
                print('Normal num {}'.format(NORMAL_NUM))
                train(args, NORMAL_NUM, generator, discriminator, optimizer_g, optimizer_d, num_images, BATCH_SIZE, seed, args.dataset)

                print('---------------------------------------------------')
                print(str(num_images), ' = size of training dataset. END.')





                print('seed {}'.format(seed))
                print('num_images {}'.format(num_images))
                print('Normal num {}'.format(NORMAL_NUM))
                print('AUC max store:', auc_max_store)
                print('F1 store:', f1_store)
                print('ACC store:', acc_store)
                auc_max_store.clear()
                f1_store.clear()
                acc_store.clear()
                print('--------------------STARTING NEXT MODEL-------------------------------')
