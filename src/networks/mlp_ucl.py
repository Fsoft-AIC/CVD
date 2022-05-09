import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bayes_layer import BayesianLinearAddNoise
from bayes_layer import BayesianLinear as BayesianLinear
from arguments import get_args
args = get_args()


class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio, unitN = 400, split = False, notMNIST=False):
        super().__init__()

        ncha, size, _ = inputsize
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla = taskcla
        self.split = split
        self.fc1 = BayesianLinear(28*28, unitN, ratio= ratio)
        self.fc2 = BayesianLinear(unitN, unitN, ratio= ratio)
        
        if not args.wo_Dropout:
            self.drop = torch.nn.Dropout(args.droprate_linear)

        # if notMNIST:
        #     self.fc3 = BayesianLinear(unitN, unitN, ratio= ratio)
        #     self.fc4 = BayesianLinear(unitN, unitN, ratio= ratio)
        self.last=torch.nn.ModuleList()
        
        if split:
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN,n))      
        else:
            self.fc3 = BayesianLinear(unitN, taskcla[0][1], ratio=ratio)
                
        
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)

        if args.wo_Dropout:
            x = F.relu(self.fc1(x, sample))
            x = F.relu(self.fc2(x, sample))
            # if self.notMNIST:
            #     x=F.relu(self.fc3(x, sample))
            #     x=F.relu(self.fc4(x, sample))
        else:
            x = self.drop(F.relu(self.fc1(x, sample)))
            x = self.drop(F.relu(self.fc2(x, sample)))
            # if self.notMNIST:
            #     x= self.drop(F.relu(self.fc3(x, sample)))
            #     x= self.drop(F.relu(self.fc4(x, sample)))
        
        if self.split:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](x))
        else:
            x = self.fc3(x, sample)
            y = F.log_softmax(x, dim=1)
        
        return y

