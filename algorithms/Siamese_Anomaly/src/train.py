import torch
from datasets.main import load_dataset
from model import *
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import evaluate, evaluate_vote
import random
import time

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, v=0.0,margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.v = v

    def forward(self, output1, vectors, feat1, label, alpha):


        euclidean_distance = torch.FloatTensor([0]).to(args.device)

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1.to(args.device), i.to(args.device))/ torch.sqrt(torch.Tensor([output1.size()[1]])).to(args.device))
        #  euclidean_distance += (F.cosine_similarity(output1, i)) / torch.sqrt(torch.Tensor([output1.size()[1]])).cuda()
          #euclidean_distance +=  torch.sum(torch.mul(output1, i)) / (torch.mul (torch.sqrt ( torch.sum( torch.square(output1) )),  torch.sqrt ( torch.sum( torch.square(i) ))      )   )

        euclidean_distance += alpha*((F.pairwise_distance(output1.to(args.device), feat1.to(args.device))) /torch.sqrt(torch.Tensor([output1.size()[1]])).to(args.device) )
        #euclidean_distance += alpha*(torch.sum(torch.mul(output1, i)) / (torch.mul (torch.sqrt ( torch.sum( torch.square(output1) )),  torch.sqrt ( torch.sum( torch.square(i) ))      )   ).cuda() )

        #calculate the margin
        marg = (len(vectors) + alpha) * self.margin

        #penalty = (1 / (torch.sum(output1) / output1.size()[1] ) )  * 100  #output values are between 0 and 1 so don't need to square
    #    euclidean_distance = euclidean_distance + penalty
        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss
    #    loss_contrastive = torch.log(euclidean_distance)
    #    loss_contrastive = ((1-label) * euclidean_distance ) + ( (label) * torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])) )
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)
    #    print(euclidean_distance)
    #    print(loss_contrastive)

        del euclidean_distance
        return loss_contrastive


def write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, pass_num, dataset, task):
    ref_images = {}
    model.eval()
    if task == 'train':
        for feat in ind:
          if (args.dataset == 'mrart' )& (args.vote == 1):
              img1, _, _, _ ,_,_= dataset.__getitem__(feat)
          else:
              img1, _, _, _ ,_= dataset.__getitem__(feat)

          if feat == base_ind:
            ref_images['images{}'.format(feat)] = feat1
          else:
            ref_images['images{}'.format(feat)] = model.forward( img1.to(args.device).float())
    else:
        gt=[]
        count = 0
        for feat in range(dataset.__len__()):
          count+=1
          if count == 2000:
                break
          if (args.dataset == 'mrart' )& (args.vote == 1):
              img1, _, labels, _,_ ,_= val_dataset.__getitem__(feat)
          else:
              img1, _, labels, _,_ = val_dataset.__getitem__(feat)
          ref_images['images{}'.format(feat)] = model.forward( img1.to(args.device).float())
          gt.append(labels)



    feat_vecs = pd.DataFrame(ref_images['images1'].detach().cpu().numpy())
    count=0
    for j in range(1, dataset.__len__()):
        count+=1
        if count == 1999:
            break
        feat_vecs = pd.concat([feat_vecs, pd.DataFrame(ref_images['images{}'.format(j)].detach().cpu().numpy())], axis =0)



    if task != 'train':

        feat_vecs = pd.concat([feat_vecs.reset_index(drop=True), pd.DataFrame(gt)], axis =1)

    model_name_temp = model_name + '_task_' + task + '_epoch_' + str(epoch+1) + '_pass_' + str(pass_num)


    feat_vecs.to_csv('./outputs/ref_vec_by_pass/class_'+str(normal_class) + '/' +model_name_temp)


def create_batches(lst, n):

    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train(model, stop_gradient, lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, freeze, smart_samp, k, eval_epoch, get_visual, augment_no, model_type, bs, num_ref_eval, num_ref_dist, anchor_dist):
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    train_aucs=[]



    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    if freeze == True:
      np.random.seed(epochs)
      rand_freeze = np.random.randint(len(indexes) )
      base_ind = ind[rand_freeze]
      feat1 = init_feat_vec(model,base_ind , train_dataset)
    else:
          feat1 = None
          base_ind = -1


    patience = 0
    max_patience = 2
    best_val_auc = 0
    best_val_auc_min = 0
    best_f1=0
    best_acc=0
    max_iter = 0
    patience2 = 5
    stop_training = False

    if augment_no > 0:
        for i in range(augment_no):

            ind.append(N+i)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        if model_type == 'RESNET':
            model.apply(deactivate_batchnorm)

        loss_sum = 0
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)

        np.random.shuffle(ind)

        batches = list(create_batches(ind, bs))



        for i in range(int(np.ceil(len(ind) / bs))):


        #for i, index in enumerate(ind):

            if (i == 0) & (get_visual == 1): #start of each epoch
                write_feat_vec(ind, indexes, base_ind , feat1, model, model_name, normal_class, epoch, i, train_dataset,'train')
                write_feat_vec(ind, indexes, base_ind , feat1, model, model_name, normal_class, epoch, i, val_dataset, 'validation')


            elif get_visual == 2: #each pass through the model
                write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, i, train_dataset, 'train')
                write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, i, val_dataset, 'validation')

            elif get_visual == 3: #every pass in the first epoch and then start of each epoch from there on
                if epoch ==0:
                    write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, i, train_dataset, 'train')
                    write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, i, val_dataset, 'validation')
                elif i ==0:
                    write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, i, train_dataset, 'train')
                    write_feat_vec(ind, indexes, base_ind, feat1, model, model_name, normal_class, epoch, i, val_dataset, 'validation')




        #    seed = (epoch+1) * (index+1)


            model.train()
            for batch_ind,index in enumerate(batches[i]):
                seed = (epoch+1) * (i+1) * (batch_ind+1)

                if (args.dataset == 'mrart' )& (args.vote == 1):
                    img1, img2, labels, base,_,_ = train_dataset.__getitem__(index, seed, base_ind)
                else:
                    img1, img2, labels, base,_ = train_dataset.__getitem__(index, seed, base_ind)



                # Forward
                img1 = img1.to(args.device)
                img2 = img2.to(args.device)
                labels = labels.to(args.device)



                if (freeze == True) & (index ==base_ind):
                  output1 = feat1
                else:
                  output1 = model.forward(img1.float())

                if (smart_samp == 0) & (k>1):

                  vecs=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, k):

                    if (ind2[j] != base_ind) & (index != ind2[j]):

                      output2=model(train_dataset.__getitem__(ind[j], seed, base_ind)[0].to(args.device).float())
                      vecs.append(output2)

                elif smart_samp == 0:

                  if (freeze == True) & (base == True):
                    output2 = feat1
                  else:
                    output2 = model.forward(img2.float())

                  vecs = [output2]

                else:
                  max_eds = [0] * k
                  max_inds = [-1] * k
                  max_ind =-1
                  vectors=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, num_ref_dist):

                    if ((num_ref_dist ==1) & (ind2[j] == base_ind)) | ((num_ref_dist ==1) & (ind2[j] == index)):
                        c = 0
                        while ((ind2[j] == base_ind) | (index == ind2[j])):
                            np.random.seed(seed * c)
                            j = np.random.randint(len(ind) )
                            c = c+1

                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(args.device).float())
                      vectors.append(output2)
                      euclidean_distance = F.pairwise_distance(output1, output2)

                      for b, vec in enumerate(max_eds):
                          if euclidean_distance > vec:
                            max_eds.insert(b, euclidean_distance)
                            max_inds.insert(b, len(vectors)-1)
                            if len(max_eds) > k:
                              max_eds.pop()
                              max_inds.pop()
                            break

                  vecs = []

                  for x in max_inds:
                      with torch.no_grad():
                          vecs.append(vectors[x])

                if batch_ind ==0:
                    loss = criterion(output1,vecs,feat1,labels,alpha)
                else:
                    loss = loss + criterion(output1,vecs,feat1,labels,alpha)

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


        train_losses.append((loss_sum / len(ind)))

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))


        if (eval_epoch == 1):
            output_name = model_name + '_output_epoch_' + str(epoch+1)
            if args.vote == 1:
                val_loss,df , ref_vecs, val_auc_min,f1,acc, val_auc, f1_score2, acc_score2 , acc_sev, acc_sev2  = evaluate_vote(args.sample, feat1, freeze, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval, anchor_dist, args.mean_dist, args.device)

            else:
                val_auc, val_loss, val_auc_min, f1, acc,df, ref_vecs, acc_sev = evaluate(feat1, freeze, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval, anchor_dist, args.mean_dist, args.device)
            train_aucs.append(val_auc_min)
            print('Validation AUC is {}'.format(val_auc))
            print("Epoch: {}, Validation loss: {}".format(epoch+1, val_loss))
            if val_auc_min > best_val_auc_min:
                best_val_auc = val_auc
                best_val_auc_min = val_auc_min
                best_epoch = epoch
                best_f1 = f1
                best_acc = acc
                best_df=df
                max_iter = 0

                model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3)) + '_min_auc_' + str(np.round(val_auc_min, 3))
                for f in os.listdir('./outputs/models/class_'+str(normal_class) + '/'):
                  if (model_name in f) :
                      os.remove(f'./outputs/models/class_'+str(normal_class) + '/{}'.format(f))
                torch.save(model.state_dict(), './outputs/models/class_'+str(normal_class)+'/' + model_name_temp)


                for f in os.listdir('./outputs/ED/class_'+str(normal_class) + '/'):
                  if (model_name in f) :
                      os.remove(f'./outputs/ED/class_'+str(normal_class) + '/{}'.format(f))
                df.to_csv('./outputs/ED/class_'+str(normal_class)+'/' +model_name_temp)

                for f in os.listdir('./outputs/ref_vec/class_'+str(normal_class) + '/'):
                  if (model_name in f) :
                    os.remove(f'./outputs/ref_vec/class_'+str(normal_class) + '/{}'.format(f))
                ref_vecs.to_csv('./outputs/ref_vec/class_'+str(normal_class) + '/' +model_name_temp)

                training_time_temp = time.time() - start_time

                if args.vote == 1:

                    cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'epoch', 'training_time','AUC_mean', 'f1_mean','acc_mean','AUC_max', 'f1_max', 'acc_max', 'sev_mean', 'sev_max']
                    params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, alpha, lr, weight_decay, args.vector_size, biases, smart_samp, k, args.v, args.contamination, epoch+1, training_time_temp,val_auc_min,f1,acc, val_auc, f1_score2, acc_score2 , acc_sev, acc_sev2 ]

                else:
                    cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time', 'f1','acc','acc_sev']
                    params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, alpha, lr, weight_decay, args.vector_size, biases, smart_samp, k, args.v, args.contamination, val_auc, epoch+1, val_auc_min, training_time_temp,f1,acc,acc_sev]
                string = './outputs/class_' + str(normal_class)
                if not os.path.exists(string):
                    os.makedirs(string)
                pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name_temp)
                pd.DataFrame(train_losses).to_csv('./outputs/losses/class_'+str(normal_class)+'/'+model_name)
                pd.DataFrame(train_aucs).to_csv('./outputs/train_aucs/class_'+str(normal_class)+'/'+model_name)



            else:
                max_iter+=1

            if max_iter == patience2:
                break


        elif epoch > 1:
          decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)
        #  print((((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100))
         # print((((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100))
          #print(decrease)

          if decrease <= 0.5:
            patience += 1


          if  (patience==max_patience) | (epoch == epochs-1):
              stop_training = True


        elif epoch == epochs-1:
            stop_training=True
        if stop_training == True:
            print("--- %s seconds ---" % (time.time() - start_time))
            training_time = time.time() - start_time
            output_name = model_name + '_output_epoch_' + str(epoch+1)
            if args.vote ==1:
                val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs,  acc_sev = evaluate_vote(args.sample, feat1,freeze, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval, anchor_dist, args.mean_dist, args.device)
            else:
                val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs,  acc_sev = evaluate(feat1,freeze, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval, anchor_dist, args.mean_dist, args.device)

            model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3)) + '_min_auc_' + str(np.round(val_auc_min, 3))
            for f in os.listdir('./outputs/models/class_'+str(normal_class) + '/'):
              if (model_name in f) :
                  os.remove(f'./outputs/models/class_'+str(normal_class) + '/{}'.format(f))
            torch.save(model.state_dict(), './outputs/models/class_'+str(normal_class)+'/' + model_name_temp)


            for f in os.listdir('./outputs/ED/class_'+str(normal_class) + '/'):
              if (model_name in f) :
                  os.remove(f'./outputs/ED/class_'+str(normal_class) + '/{}'.format(f))
            df.to_csv('./outputs/ED/class_'+str(normal_class)+'/' +model_name_temp)

            for f in os.listdir('./outputs/ref_vec/class_'+str(normal_class) + '/'):
              if (model_name in f) :
                os.remove(f'./outputs/ref_vec/class_'+str(normal_class) + '/{}'.format(f))
            ref_vecs.to_csv('./outputs/ref_vec/class_'+str(normal_class) + '/' +model_name_temp)


             #write out all details of model training
            cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time', 'f1','acc','acc_sev']
            params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, alpha, lr, weight_decay, args.vector_size, biases, smart_samp, k, args.v, args.contamination, val_auc, epoch+1, val_auc_min, training_time,f1,acc,acc_sev]
            string = './outputs/class_' + str(normal_class)
            if not os.path.exists(string):
                os.makedirs(string)
            pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name_temp)
            pd.DataFrame(train_losses).to_csv('./outputs/losses/class_'+str(normal_class)+'/'+model_name)


            break



    print("Finished Training")
    if eval_epoch == 1:
        print("AUC was {} on epoch {}".format(best_val_auc, best_epoch))
        return best_val_auc, best_epoch, best_val_auc_min, training_time_temp, best_f1, best_acc,train_losses
    else:
        print("AUC was {} on epoch {}".format(val_auc, epoch))
        return val_auc, epoch, val_auc_min, training_time, f1,acc, train_losses






def init_feat_vec(model,base_ind, train_dataset ):

        model.eval()
        if (args.dataset == 'mrart' )& (args.vote == 1):
            feat1,_,_,_ ,_,_= train_dataset.__getitem__(base_ind)
        else:
            feat1,_,_,_ ,_= train_dataset.__getitem__(base_ind)



        with torch.no_grad():
          feat1 = model(feat1.to(args.device).float()).to(args.device)

        return feat1



def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class,task, data_path,download_data= download_data) #get all training data
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0] #get indexes in the training set that are equal to the normal class
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N) #randomly sample N normal data points
    final_indexes = ind[samp]
    if contamination != 0:
      numb = np.ceil(N*contamination)
      if numb == 0.0:
        numb=1.0

      con = np.where(np.array(train_dataset.targets)!=normal_class)[0] #get indexes of non-normal class
      samp = random.sample(range(0, len(con)), int(numb))
      samp2 = random.sample(range(0, len(final_indexes)), len(final_indexes) - int(numb))
      final_indexes = np.array(list(final_indexes[samp2]) + list(con[samp]))
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 1)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--freeze', default = True)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--get_visual', choices = [0,1,2,3], type=int, default=0)
    parser.add_argument('--augment_no', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--stop_gradient', type=int, default=1)
    parser.add_argument('--num_ref_dist', type=int, default=None)
    parser.add_argument('--anchor_dist', type=int, default=0)
    parser.add_argument('--mean_dist', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--vote', type=int, default=0)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--data_split_path', type=str, default=None)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    model_name = args.model_name
    model_type = args.model_type
    dataset_name = args.dataset
    normal_class = args.normal_class
    N = args.num_ref
    seed = args.seed
    anchor_dist = args.anchor_dist
    freeze = args.freeze
    epochs = args.epochs
    data_path = args.data_path
    download_data = args.download_data
    contamination = args.contamination
    indexes = args.index
    alpha = args.alpha
    lr = args.lr
    vector_size = args.vector_size
    weight_decay = args.weight_decay
    smart_samp = args.smart_samp
    k = args.k
    weight_init_seed = args.weight_init_seed
    v = args.v
    task = args.task
    eval_epoch = args.eval_epoch
    get_visual = args.get_visual
    augment_no = args.augment_no
    bs = args.batch_size
    biases = args.biases
    num_ref_eval = args.num_ref_eval
    num_ref_dist = args.num_ref_dist
    stop_gradient = args.stop_gradient
    if num_ref_eval == None:
        num_ref_eval = N
    if num_ref_dist == None:
        num_ref_dist = N


    if dataset_name == 'mrart':
        normal_class =1

    #if indexes for reference set aren't provided, create the reference set.
    if (dataset_name != 'mvtec') & (dataset_name != 'ixi')  :
        if  (args.vote == 0):
            if indexes != []:
                indexes = [int(item) for item in indexes.split(', ')]
            else:
                indexes = create_reference(contamination, dataset_name, normal_class, 'train', data_path, download_data, N, seed)

    #create train and test set
    if (dataset_name =='mvtec') | (dataset_name =='ixi') :
        train_dataset = load_dataset(dataset_name, indexes, normal_class, 'train',  data_path, download_data, seed, N=N, sample=args.sample,data_split_path=args.data_split_path)
        indexes = train_dataset.indexes
    elif ((dataset_name == 'mrart') & (args.vote ==1)):
        train_dataset = load_dataset(dataset_name+'_vote', indexes, normal_class, 'train',  data_path, download_data, seed, N=N, sample=args.sample,data_split_path=args.data_split_path)
        indexes = train_dataset.indexes
    else:

        print(args.data_split_path)
        train_dataset = load_dataset(dataset_name, indexes, normal_class, 'train',  data_path, download_data = download_data,seed=seed, N=N, sample=args.sample,data_split_path=args.data_split_path)


    if task != 'train':
        if ((dataset_name == 'mrart') & (args.vote ==1)):
            val_dataset = load_dataset(dataset_name+'_vote', indexes,  normal_class, 'test', data_path, download_data=False, seed=seed,N=N, sample=args.sample,data_split_path=args.data_split_path)
        else:
            val_dataset = load_dataset(dataset_name, indexes,  normal_class, 'test', data_path, download_data=False, seed=seed, N=N, sample=args.sample,data_split_path=args.data_split_path)
    else:
        val_dataset = load_dataset(dataset_name, indexes, normal_class, 'validate', data_path, download_data=False)




    #set the seed
    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)

    #create directories
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')

    string = './outputs/models/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    if not os.path.exists('outputs/ED'):
        os.makedirs('outputs/ED')

    string = './outputs/ED/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/ref_vec'):
        os.makedirs('outputs/ref_vec')

    string = './outputs/ref_vec/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/losses'):
        os.makedirs('outputs/losses')

    string = './outputs/losses/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/train_aucs'):
        os.makedirs('outputs/train_aucs')

    string = './outputs/train_aucs/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/ref_vec_by_pass/'):
        os.makedirs('outputs/ref_vec_by_pass')

    string = './outputs/ref_vec_by_pass/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/inference_times'):
        os.makedirs('outputs/inference_times')
    if not os.path.exists('outputs/inference_times/class_' + str(normal_class)):
        os.makedirs('outputs/inference_times/class_'+str(normal_class))



    #Initialise the model
    if model_type == 'MVTEC_VGG3':
            model = MVTEC_VGG3_pre(vector_size, biases)
    if model_type == 'CIFAR_VGG3':
        if args.pretrain == 1:
            model = CIFAR_VGG3_pre(vector_size, biases)
        else:
            model = CIFAR_VGG3(vector_size, biases)
    elif model_type == 'MNIST_VGG3':
        if args.pretrain == 1:
            model = MNIST_VGG3_pre(vector_size, biases)
        else:
            model = MNIST_VGG3(vector_size, biases)
    elif model_type == 'MNIST_LENET':
        model = MNIST_LeNet(vector_size, biases)
    elif model_type == 'RESNET':
        model = RESNET_pre(vector_size, biases)
    elif model_type == 'RESNET_batch':
        model =resnet_batch()
    elif model_type == 'CIFAR_LENET':
        model = CIFAR_LeNet(vector_size, biases)
    elif (model_type == 'CIFAR_VGG4'):
        if (args.pretrain ==1):
            model = CIFAR_VGG4_pre(vector_size, biases)
        else:
            model = CIFAR_VGG4(vector_size, biases)

    elif (model_type == 'FASHION_VGG3'):
        if (args.pretrain ==1):
            model = FASHION_VGG3_pre(vector_size, biases)
        else:
            model = FASHION_VGG3(vector_size, biases)
    elif (model_type == 'ALEXNET'):
        if (args.pretrain ==1):
            if args.vote == 1:
                model =ALEXNET_nomax_pre()
            else:
                model =ALEXNET_pre()




    if (model_type == 'RESNET'):
        model.apply(deactivate_batchnorm)

    model_name = model_name + '_normal_class_' + str(normal_class) + '_seed_' + str(seed)
    criterion = ContrastiveLoss(v)
    auc, epoch, auc_min, training_time, f1,acc, train_losses= train(model,stop_gradient,lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, freeze, smart_samp,k, eval_epoch, get_visual, augment_no, model_type, bs, num_ref_eval, num_ref_dist, anchor_dist)
