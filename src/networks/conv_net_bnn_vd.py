import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args

args = get_args()
if args.drop_prior == 'gauss':
    from dropout.Gauss_dropout import GaussDropoutConv2d as DropoutConv2d
    from dropout.Gauss_dropout import GaussDropout as DropoutLinear
elif args.drop_prior == 'log_uniform':
    from dropout.LUGauss_dropout import GaussDropoutConv2d as DropoutConv2d
    from dropout.LUGauss_dropout import GaussDropout as DropoutLinear
    
if args.approach == 'ucl_vd':
    from bayes_layer import BayesianLinear as BayesianLinear
    from bayes_layer import BayesianConv2D as BayesianConv2D
elif args.approach == 'vcl_vd':    
    from bayes_layer import BayesianLinearVCL_bias as BayesianLinear
    from bayes_layer import BayesianConv2DVCL as BayesianConv2D
    


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        

        # if args.conv_Dropout:
        #     self.drop1 = DropoutConv2d(in_channels=ncha, p= 0.2, size= size)
        self.conv1 = BayesianConv2D(ncha,32,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(size,3, padding=1) # 32

        # if args.conv_Dropout:
        #     self.drop2 = DropoutConv2d(in_channels=32, p= args.droprate, size= s)
        self.conv2 = BayesianConv2D(32,32,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16

        if args.conv_Dropout:
            self.drop3 = DropoutConv2d(in_channels=32, p= args.droprate, size= s, clip=args.clip)
        self.conv3 = BayesianConv2D(32,64,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 16

        # if args.conv_Dropout:
        #     self.drop4 = DropoutConv2d(in_channels=64, p= args.droprate, size= s)
        self.conv4 = BayesianConv2D(64,64,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8

        if args.conv_Dropout:
            self.drop5 = DropoutConv2d(in_channels=64, p= args.droprate, size= s, clip=args.clip)
        self.conv5 = BayesianConv2D(64,128,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 8

        # if args.conv_Dropout:
        #     self.drop6 = DropoutConv2d(in_channels=128, p= args.droprate, size= s)
        self.conv6 = BayesianConv2D(128,128,kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        if args.conv_Dropout:
            self.drop6 = DropoutConv2d(in_channels=128, p= args.droprate, size= s,clip=args.clip)

        # self.drop7 = DropoutLinear(input_size=s*s*128, p=args.droprate)
        self.fc1 = BayesianLinear(s*s*128,256, ratio = ratio)
        self.drop7 = DropoutLinear(input_size=256, p=args.droprate,clip=args.clip)

        if not args.conv_Dropout:     
            self.drop = nn.Dropout(0.25)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample=False, noise=None):
        kld = []
        if args.conv_Dropout: # DROPOUT ON CONVs
            if self.training:
                # h, kl = self.drop1(x)
                # kld.append(kl)
                h = self.relu(self.conv1(x, sample))

                # h, kl = self.drop2(h)
                # kld.append(kl)
                h = self.relu(self.conv2(h, sample))
                h = self.MaxPool(h)
                
                h, kl = self.drop3(h)
                kld.append(kl)
                h = self.relu(self.conv3(h, sample))

                # h, kl = self.drop4(h)
                # kld.append(kl)
                h = self.relu(self.conv4(h, sample))
                h = self.MaxPool(h)

                h, kl = self.drop5(h)
                kld.append(kl)
                h = self.relu(self.conv5(h, sample))

                # h, kl = self.drop6(h)
                # kld.append(kl)
                h = self.relu(self.conv6(h, sample))
                h = self.MaxPool(h)
                h, kl = self.drop6(h)
                kld.append(kl)

                h = h.view(h.shape[0],-1)
                # h, kl = self.drop7(h)
                # kld.append(kl)
                h = self.relu(self.fc1(h, sample))
                h, kl = self.drop7(h)
                kld.append(kl)
            else:
                # h, kl = self.drop1(x, noise['drop1.log_alpha'])
                # kld.append(kl) 
                h = self.relu(self.conv1(x, sample))

                # h, kl = self.drop2(h, noise['drop2.log_alpha'])
                # kld.append(kl) 
                h = self.relu(self.conv2(h, sample))
                h = self.MaxPool(h)

                if args.drop_prior == 'gauss':
                  h, kl = self.drop3(h, [noise['drop3.muy'], noise['drop3.log_alpha']])
                else :
                  h, kl = self.drop3(h, noise['drop3.log_alpha'])
                kld.append(kl)
                h = self.relu(self.conv3(h, sample))
                
                # h, kl = self.drop4(h, noise['drop4.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.conv4(h, sample))
                h = self.MaxPool(h)

                if args.drop_prior == 'gauss':
                  h, kl = self.drop5(h, [noise['drop5.muy'], noise['drop5.log_alpha']])
                else :
                  h, kl = self.drop5(h, noise['drop5.log_alpha'])
                kld.append(kl)
                h = self.relu(self.conv5(h, sample))

                # h, kl = self.drop6(h, noise['drop6.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.conv6(h, sample))
                h = self.MaxPool(h)
                if args.drop_prior == 'gauss':
                  h, kl = self.drop6(h, [noise['drop6.muy'], noise['drop6.log_alpha']])
                else :
                  h, kl = self.drop6(h, noise['drop6.log_alpha'])
                kld.append(kl) 

                h = h.view(h.shape[0],-1)

                # h, kl = self.drop7(h, noise['drop7.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.fc1(h, sample))
                if args.drop_prior == 'gauss':
                  h, kl = self.drop7(h, [noise['drop7.muy'], noise['drop7.log_alpha']])
                else :
                  h, kl = self.drop7(h, noise['drop7.log_alpha'])
                kld.append(kl)
        else: #  BASIC DROPOUT ON CONVs
            h=self.relu(self.conv1(x,sample))
            h=self.relu(self.conv2(h,sample))
            h=self.drop(self.MaxPool(h))
            h=self.relu(self.conv3(h,sample))
            h=self.relu(self.conv4(h,sample))
            h=self.drop(self.MaxPool(h))
            h=self.relu(self.conv5(h,sample))
            h=self.relu(self.conv6(h,sample))
            h=self.drop(self.MaxPool(h))

            h=h.view(h.shape[0], -1)
            if self.training:
                # h, kl = self.drop7(h)
                # kld.append(kl)
                h = self.relu(self.fc1(h, sample))
                h, kl = self.drop7(h)
                kld.append(kl)
            else: 
                # h, kl = self.drop7(h, noise['drop7.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.fc1(h, sample))
                if args.drop_prior == 'gauss':
                  h, kl = self.drop7(h, [noise['drop7.muy'], noise['drop7.log_alpha']])
                else :
                  h, kl = self.drop7(h, noise['drop7.log_alpha'])
                kld.append(kl)

        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        if self.training:
            return y, sum(kld)
        return y
        
