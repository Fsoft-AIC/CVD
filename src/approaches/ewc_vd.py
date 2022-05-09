import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
# from torchvision import models
# from torchvision.models.resnet import *
args = get_args()


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, nepochs= 100, sbatch= 256, lr= 0.001, lr_min= 1e-6, lr_factor= 3, lr_patience= 5, clipgrad= 100, args= None, log_name= None, 
                split= False):
        self.model = model
        self.model_old = model
        self.fisher = None

        file_name = log_name
        self.logger = utils.logger(file_name= file_name, resume= False, path= './result_data/csvdata/', data_format= 'csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.split = split
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.lamb = args.lamb
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
        
        self.noise = {}
        self.valid = True

        return

    def _get_optimizer(self, lr= None):
        if lr is None: 
            lr = self.lr
        
        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr= lr)
        if args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr= lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        self.valid = True
        noise = {}
        for name, param in self.model.named_parameters():
            if "alpha" in name or "muy" in name:
                noise[name] = torch.zeros(param.data.size())
        self.noise[t] = noise

        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        datasize = xtrain.size(0)
        if args.KL_coeff == '1':
            self.KL_coeff = 1
        elif args.KL_coeff == '1_M':
            self.KL_coeff = 1/self.sbatch
        elif args.KL_coeff == '1_N':
            self.KL_coeff = 1/datasize
        elif args.KL_coeff == 'M_N':
            self.KL_coeff = self.sbatch/datasize

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            
            num_batch = xtrain.size(0)
            
            self.train_epoch(t, xtrain, ytrain)
            
            clock1 = time.time()
                       
            if (e+1) % 20 == 0:
                train_loss, train_acc = self.eval(t, xtrain, ytrain)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                    e+1, 1000 *self.sbatch *(clock1-clock0) / num_batch,
                    1000*self.sbatch*(clock2-clock1)/num_batch, train_loss, 100*train_acc),end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            if (e+1) % 20 == 0:
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100*valid_acc), end='')
                print()
            
                #save log for current task & old tasks at every epoch
                self.logger.add(epoch= (t * self.nepochs) + e, task_num= t+1, valid_loss= valid_loss, valid_acc= valid_acc)

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                # print(' *', end='')
            
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    # print(' lr={:.1e}'.format(lr), end='')
                    # if lr < self.lr_min:
                    #     # print()
                    #     if args.conv_net:
                    #         pass
                    #         break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            

        # Restore best
        utils.set_model_(self.model, best_model)

        self.logger.save()
        
        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t > 0:
            fisher_old = {}
            for n, _ in self.model.named_parameters():
                if "alpha" in n or "muy" in n:
                    continue
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = utils.fisher_matrix_diag(t, xtrain, ytrain, self.model, self.criterion, split = self.split)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                if "alpha" in n or "muy" in n:
                    continue
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (t+1)
        self.valid = False

        return

    def train_epoch(self, t, x, y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r): 
                b = r[i:i+self.sbatch]
            else: 
                b = r[i:]
            images = x[b]
            targets = y[b]
            # Forward current model
            avg_loss = 0
            for _ in range(args.num_samples):
                if self.split:
                    outputs, kld = self.model.forward(images, True)
                    outputs = outputs[t]
                else:
                    if args.multi_head:
                        outputs, kld = self.model.forward(images, True)
                        outputs = outputs[t]        
                    else:              
                        outputs, kld = self.model.forward(images, True)

                likelihood = self.criterion(t, outputs, targets)
                loss = likelihood + kld * self.KL_coeff * args.KL_weight
                avg_loss += loss
            avg_loss = avg_loss/args.num_samples

            # Backward
            self.optimizer.zero_grad()
            avg_loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        # Update noise for task t
        noise = {}
        for name, param in self.model.named_parameters():
            if "alpha" in name or "muy" in name:
                noise[name] = param.data.clone().detach()
                # print(name)
        self.noise[t] = noise

        return

    def eval(self,t,x,y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()
        if self.valid:
            num_samples = 1
        else:
            num_samples = args.test_samples
        # Loop batches
        with torch.no_grad():
            for i in range(0, len(r), self.sbatch):
                if i + self.sbatch <= len(r): 
                    b = r[i:i+self.sbatch]
                else: 
                    b= r[i:]
                images = x[b]
                targets = y[b]
                
                # Forward 
                avg_output = 0
                for _ in range(num_samples):
                    if self.split:
                        output = self.model.forward(images, noise= self.noise[t])[t]
                    else:
                        output = self.model.forward(images, noise= self.noise[t])
                    avg_output += output/num_samples
                    
                loss = self.criterion(t, avg_output, targets)
                _, pred = avg_output.max(1)
                hits = (pred == targets).float()

                total_loss += loss.data.cpu().numpy() * len(b)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(b)

        return total_loss/total_num, total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t > 0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                if "alpha" in name or "muy" in name:
                    continue
                loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2)) / 2
        loss = self.ce(output, targets) + self.lamb * loss_reg
        return loss