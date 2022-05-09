import sys
import torch
import torch.nn as nn
from utils import *
from arguments import get_args

args = get_args()
if args.drop_prior == 'gauss':
    from dropout.Gauss_dropout import GaussDropoutConv2d as DropoutConv2d
    from dropout.Gauss_dropout import GaussDropout as DropoutLinear
elif args.drop_prior == 'log_uniform':
    from dropout.LUGauss_dropout import GaussDropoutConv2d as DropoutConv2d
    from dropout.LUGauss_dropout import GaussDropout as DropoutLinear

class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        

        # if args.conv_Dropout:
        #     self.drop1 = DropoutConv2d(in_channels=ncha, p= args.droprate, size= size)
        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32

        # if args.conv_Dropout:
        #     self.drop2 = DropoutConv2d(in_channels=32, p= args.droprate, size= s)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16

        if args.conv_Dropout:
            self.drop3 = DropoutConv2d(in_channels=32, p= args.droprate, size= s, clip=args.clip)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        
        # if args.conv_Dropout:
        #     self.drop4 = DropoutConv2d(in_channels=64, p= args.droprate, size= s)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        
        if args.conv_Dropout:
            self.drop5 = DropoutConv2d(in_channels=64, p= args.droprate, size= s, clip=args.clip)
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        
        # if args.conv_Dropout:
        #     self.drop6 = DropoutConv2d(in_channels=128, p= args.droprate, size= s)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        if args.conv_Dropout:
            self.drop6 = DropoutConv2d(in_channels=128, p= args.droprate, size= s, clip=args.clip)
        if not args.conv_Dropout:     
            self.drop = nn.Dropout(args.droprate_linear)

        # self.drop_test = DropoutConv2d(in_channels=128, p= args.droprate, size= s)
        # self.drop7 = DropoutLinear(input_size=s*s*128, p=args.droprate)
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop7 = DropoutLinear(input_size=256, p=args.droprate, clip=args.clip)

        self.MaxPool = torch.nn.MaxPool2d(2)       
        self.last = torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256, n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, train= False, noise= None):
        kld = []
        # #################### VD ON CONV
        if args.conv_Dropout:
            if self.training:
                # h, kl = self.drop1(x)
                # kld.append(kl)
                h = self.relu(self.conv1(x))

                # h, kl = self.drop2(h)
                # kld.append(kl)
                h = self.relu(self.conv2(h))
                h = self.MaxPool(h)
                
                h, kl = self.drop3(h)
                kld.append(kl)
                h = self.relu(self.conv3(h))

                # h, kl = self.drop4(h)
                # kld.append(kl)
                h = self.relu(self.conv4(h))
                h = self.MaxPool(h)

                h, kl = self.drop5(h)
                kld.append(kl)
                h = self.relu(self.conv5(h))

                # h, kl = self.drop6(h)
                # kld.append(kl)
                h = self.relu(self.conv6(h))
                h = self.MaxPool(h)
                h, kl = self.drop6(h)
                kld.append(kl)

                h = h.view(h.shape[0],-1)
                # h, kl = self.drop7(h)
                # kld.append(kl)
                h = self.relu(self.fc1(h))
                h, kl = self.drop7(h)
                kld.append(kl)
            else:
                # h, kl = self.drop1(x, noise['drop1.log_alpha'])
                # kld.append(kl) 
                h = self.relu(self.conv1(x))

                # h, kl = self.drop2(h, noise['drop2.log_alpha'])
                # kld.append(kl) 
                h = self.relu(self.conv2(h))
                h = self.MaxPool(h)

                if args.drop_prior == 'gauss':
                  h, kl = self.drop3(h, [noise['drop3.muy'], noise['drop3.log_alpha']])
                else :
                  h, kl = self.drop3(h, noise['drop3.log_alpha'])
                kld.append(kl)
                h = self.relu(self.conv3(h))
                
                # h, kl = self.drop4(h, noise['drop4.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.conv4(h))
                h = self.MaxPool(h)

                if args.drop_prior == 'gauss':
                  h, kl = self.drop5(h, [noise['drop5.muy'], noise['drop5.log_alpha']])
                else :
                  h, kl = self.drop5(h, noise['drop5.log_alpha'])
                kld.append(kl)
                h = self.relu(self.conv5(h))

                # h, kl = self.drop6(h, noise['drop6.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.conv6(h))
                h = self.MaxPool(h)
                if args.drop_prior == 'gauss':
                  h, kl = self.drop6(h, [noise['drop6.muy'], noise['drop6.log_alpha']])
                else :
                  h, kl = self.drop6(h, noise['drop6.log_alpha'])
                kld.append(kl)
                h = h.view(h.shape[0],-1)

                # h, kl = self.drop7(h, noise['drop7.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.fc1(h))
                if args.drop_prior == 'gauss':
                  h, kl = self.drop7(h, [noise['drop7.muy'], noise['drop7.log_alpha']])
                else :
                  h, kl = self.drop7(h, noise['drop7.log_alpha'])
                kld.append(kl)
        else:  ################## BASIC DROPOUT ON CONV    
            h = self.relu(self.conv1(x))
            h = self.relu(self.conv2(h))
            h = self.drop(self.MaxPool(h))
            h = self.relu(self.conv3(h))
            h = self.relu(self.conv4(h))
            h = self.drop(self.MaxPool(h))
            h = self.relu(self.conv5(h))
            h = self.relu(self.conv6(h))
            h = self.drop(self.MaxPool(h))
            # h, kl = self.drop_test(h)
            # kld.append(kl)
            h = h.view(h.shape[0],-1)
            if self.training:
                # h, kl = self.drop7(h)
                # kld.append(kl)
                h = self.relu(self.fc1(h))
                h, kl = self.drop7(h)
                kld.append(kl)
            else: 
                # h, kl = self.drop7(h, noise['drop7.log_alpha'])
                # kld.append(kl)
                h = self.relu(self.fc1(h))
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