import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

args = get_args()
if args.approach == 'ucl':
    from bayes_layer import BayesianConv2D
    from bayes_layer import BayesianLinear
elif args.approach == 'vcl':
    from bayes_layer import BayesianConv2DVCL as BayesianConv2D
    from bayes_layer import BayesianLinearVCL_bias as BayesianLinear

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,32,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = BayesianConv2D(32,32,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = BayesianConv2D(32,64,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = BayesianConv2D(64,64,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = BayesianConv2D(64,128,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = BayesianConv2D(128,128,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = BayesianLinear(s*s*128, 256, ratio = ratio)
        if not args.wo_Dropout:
            self.drop1 = nn.Dropout(0.25)
            self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample= False):
        h = self.relu(self.conv1(x,sample))
        h = self.relu(self.conv2(h,sample))
        h = self.MaxPool(h)
        if not args.wo_Dropout:
            h = self.drop1(h)
        h = self.relu(self.conv3(h,sample))
        h = self.relu(self.conv4(h,sample))
        h = self.MaxPool(h)
        if not args.wo_Dropout:
            h = self.drop1(h)
        h = self.relu(self.conv5(h,sample))
        h = self.relu(self.conv6(h,sample))
        h = self.MaxPool(h)
        if not args.wo_Dropout:
            h = self.drop1(h)
        h = h.view(h.shape[0], -1)
        h = self.relu(self.fc1(h, sample))
        if not args.wo_Dropout:
            h = self.drop2(h)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y
        
