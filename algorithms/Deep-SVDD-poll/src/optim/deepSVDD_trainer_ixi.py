from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import pandas as pd
import logging
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DeepSVDDTrainer_ixi(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, eval_epoch:int =0,  early_stopping_loss: int=0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.eval_epoch = eval_epoch

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_auc2 = None
        self.test_time = None
        self.test_scores = None
        self.final_data=None

    def train(self, dataset: BaseADDataset, net: BaseNet, eval_epoch):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader


        train_loader = DataLoader(dataset=dataset[0], batch_size=self.batch_size, shuffle=True,
                                  num_workers=0)




        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        best_auc=0
        pat=0
        max_pat=5
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _,_ ,_= data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

            if eval_epoch ==1:
                df, test_auc=self.test(dataset, net,  eval_epoch, epoch=epoch)
                if test_auc > best_auc:
                    self.final_data = df
                    best_auc=test_auc
                    pat=0
                else:
                    pat+=1

                if pat == max_pat:
                    print('Early stopping: best AUC was {}'.format(best_auc))
                    break



        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        print('in optim')
        print(self.final_data)
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, eval_epoch =0, epoch=None):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader

        #test_loader = DataLoader(dataset=dataset[1], batch_size=self.batch_size, shuffle=False,
        #                                                    num_workers=0)
        test_loader = DataLoader(dataset=dataset[1], batch_size=1, shuffle=False,
                                                            num_workers=0)
        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        dict_score1={}
        dict_label={}
        dict_label_sev={}
        used = []
        orig_files=[]
        inf_times=[]
        with torch.no_grad():
            for c,data in enumerate(test_loader):
                score_temp = []
                inputs, labels, idx, label_sev, original_file = data
                inputs = inputs.to(self.device)
                t1=time.time()
                outputs = net(inputs)

                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                inf_times.append(time.time() - t1)
                for i in range(len(labels)):
        #            print(labels[i])
        #            print(labels[i].cpu().data.numpy().tolist())
                    if original_file[i] in used:
                        dict_score1['{}'.format(original_file[i])].append(scores[i].cpu().data.numpy())
                        if (labels[i].cpu().data.numpy().tolist() == 1) & (dict_label['{}'.format(original_file[i])] ==0):
                            print(original_file[i])
                        dict_label['{}'.format(original_file[i])]=labels[i].cpu().data.numpy().tolist()
                        dict_label_sev['{}'.format(original_file[i])]=label_sev[i].cpu().data.numpy().tolist()
                        orig_files.append(original_file[i])
                    else:
                        dict_score1['{}'.format(original_file[i])] = []
                        dict_score1['{}'.format(original_file[i])].append(scores[i].cpu().data.numpy())
                    #    dict_label['{}'.format(original_file[i])]=[]
                    #    dict_label['{}'.format(original_file[i])].append(labels[i].cpu().data.numpy().tolist())
                        dict_label['{}'.format(original_file[i])]=labels[i].cpu().data.numpy().tolist()
                        dict_label_sev['{}'.format(original_file[i])] = label_sev[i].cpu().data.numpy().tolist()
                        orig_files.append(original_file[i])
                        used.append(original_file[i])



        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

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
        df2.to_csv('data2.csv')







        labels = np.array(df2['label'])
        label_sev = np.array(df2['label_sev'])
        scores1 = np.array(df2['s1'])
        scores2 = np.array(df2['s2'])

        print(df2['label'].value_counts())
        assert df2['label'].value_counts().values[0] == df2['label'].value_counts().values[1]


        #get auc and f1 based on score 1, binary
        self.test_auc = roc_auc_score(labels, scores1)
        perc_thres = int((len(df2['label'].loc[df2['label'] ==0]) / (len(df2['label'].loc[df2['label'] ==1]) + len(df2['label'].loc[df2['label'] ==0])))*100)
        print(perc_thres)
        thresh = np.percentile(scores1, perc_thres)
        y_pred = np.where(scores1 >= thresh, 1, 0)
        prec, recall, test_metric, _ = precision_recall_fscore_support(
            labels, y_pred, average="binary")
        self.test_f1 = test_metric

        #get auc and f1 based on score 2, binary
        self.test_auc2 = roc_auc_score(labels, scores2)
        thresh = np.percentile(scores2, perc_thres)
        y_pred2 = np.where(scores2 >= thresh, 1, 0)
        prec2, recall2, test_metric2, _ = precision_recall_fscore_support(
            labels, y_pred2, average="binary")
        self.test_f1_2 = test_metric2






        df=pd.concat([pd.DataFrame(scores1),pd.DataFrame(scores2), pd.DataFrame(labels), pd.DataFrame(label_sev), pd.DataFrame(y_pred), pd.DataFrame(y_pred2)], axis =1)
        df.columns = ['output', 'output2','label', 'label_sev','pred','pred2']
        print('AUC is {}'.format(roc_auc_score(labels, scores1)))
        print('prec is {}'.format(prec))
        print('recall is {}'.format(recall))
        print('AUC based on max is {}'.format(roc_auc_score(labels, scores2)))
        print('prec based on max is is {}'.format(prec2))
        print('recall based on max is is {}'.format(recall2))
        sense = len(df.loc[(df['label'] == 1) & (df['pred'] == 1)] ) /( len(df.loc[(df['label'] == 1) & (df['pred'] == 0)] ) + len(df.loc[(df['label'] == 1) & (df['pred'] == 1)] ))
        spec = len(df.loc[(df['label'] == 0) & (df['pred'] == 0)] ) /( len(df.loc[(df['label'] == 0) & (df['pred'] == 0)] ) + len(df.loc[(df['label'] == 0) & (df['pred'] == 1)] ))
        sense2 = len(df.loc[(df['label'] == 1) & (df['pred2'] == 1)] ) /( len(df.loc[(df['label'] == 1) & (df['pred2'] == 0)] ) + len(df.loc[(df['label'] == 1) & (df['pred2'] == 1)] ))
        spec2 = len(df.loc[(df['label'] == 0) & (df['pred2'] == 0)] ) /( len(df.loc[(df['label'] == 0) & (df['pred2'] == 0)] ) + len(df.loc[(df['label'] == 0) & (df['pred2'] == 1)] ))


        self.final_data = df

        self.test_acc = (sense+spec)/2
        self.test_acc2= (sense2+spec2)/2

        self.test_scores = scores1



        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test set F1: {:.2f}%'.format(100. * self.test_f1))
        logger.info('Test set Balanced Accuarcy: {:.2f}%'.format(100. * self.test_acc))

        logger.info('Test set based on min AUC: {:.2f}%'.format(100. * self.test_auc2))
        logger.info('Test set based on min F1: {:.2f}%'.format(100. * self.test_f1_2))
        logger.info('Test set based on min Balanced Accuarcy: {:.2f}%'.format(100. * self.test_acc2))

    

    #    logger.info('Severity AUC {:.2f}%'.format(100. * auc_sev ))
    #    logger.info('Severity AUC based on max {:.2f}%'.format(100. * auc_sev2 ))

        logger.info('Average inference time per batch {}'.format(np.mean(inf_times)))

        logger.info('Finished testing.')

        return df, self.test_auc

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _,_,_ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
