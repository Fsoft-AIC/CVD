import sys
import torch
import torch.nn.functional as F
from arguments import get_args

args = get_args()
if args.drop_prior == 'gauss':
    from dropout.Gauss_dropout import GaussDropoutConv2d as DropoutConv2d
    from dropout.Gauss_dropout import GaussDropout as DropoutLinear
elif args.drop_prior == 'log_uniform':
    from dropout.LUGauss_dropout import GaussDropoutConv2d as DropoutConv2d
    from dropout.LUGauss_dropout import GaussDropout as DropoutLinear

class Net(torch.nn.Module):

    def __init__(self, inputsize, taskcla, unitN = 400, split = False, notMNIST = False):
        super(Net,self).__init__()

        ncha, size, _ = inputsize
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla = taskcla
        self.split = split
        self.relu = torch.nn.ReLU()
        # self.drop = torch.nn.Dropout(0.5)
        self.drop1 = DropoutLinear(ncha*size*size, p=args.droprate,clip=args.clip)
        self.fc1 = torch.nn.Linear(ncha*size*size, unitN)
        self.drop2 = DropoutLinear(unitN, p= args.droprate, clip=args.clip)
        self.fc2 = torch.nn.Linear(unitN, unitN)
        
        # if notMNIST:
        #     self.fc3 = torch.nn.Linear(unitN, unitN)
        #     self.fc4 = torch.nn.Linear(unitN, unitN)
        
        if split:
            self.last = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN, n))
        else:
            self.drop3 = DropoutLinear(unitN, p= args.droprate, clip=args.clip)
            self.fc3 = torch.nn.Linear(unitN, taskcla[0][1])

    def forward(self, x, train= False, noise= None):
        h = x.view(x.size(0), -1)
        kld = []

        if train:
            h, kl = self.drop1(h)     
            kld.append(kl)
            h, kl = self.drop2(F.relu(self.fc1(h)))
            kld.append(kl)
            
            h = F.relu(self.fc2(h))
            # if self.notMNIST:
            #     h = self.drop(F.relu(self.fc3(h)))
            #     h = self.drop(F.relu(self.fc4(h)))
            
            if self.split:
                y = []
                for t,i in self.taskcla:
                    y.append(self.last[t](h))
            else:
                h, kl = self.drop3(h)
                kld.append(kl)
                y = self.fc3(h)
            return y, sum(kld)
            
        else:
            if args.drop_prior == 'gauss':
              h, kl = self.drop1(h, [noise['drop1.muy'], noise['drop1.log_alpha']])
            else :
              h, kl = self.drop1(h, noise['drop1.log_alpha'])                 
            kld.append(kl)
            h = F.relu(self.fc1(h))
            if args.drop_prior == 'gauss':
              h, kl = self.drop2(h, [noise['drop2.muy'], noise['drop2.log_alpha']])
            else :
              h, kl = self.drop2(h, noise['drop2.log_alpha'])
            kld.append(kl)
            
            h = F.relu(self.fc2(h))
            # if self.notMNIST:
            #     h = self.drop(F.relu(self.fc3(h)))
            #     h = self.drop(F.relu(self.fc4(h)))
            
            if self.split:
                y = []
                for t,i in self.taskcla:
                    y.append(self.last[t](h))
            else:
                if args.drop_prior == 'gauss':
                  h, kl = self.drop3(h, [noise['drop3.muy'], noise['drop3.log_alpha']])
                else :
                  h, kl = self.drop3(h, noise['drop3.log_alpha'])
                kld.append(kl)
                y = self.fc3(h)
            return y


