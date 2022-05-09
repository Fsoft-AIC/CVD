import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter



class GaussDropout(nn.Module):
    def __init__(self, input_size, p=0.5, clip=True):
        """
            Variational Dropout
            :param input_size: An int of input size
            :param p: An initial variance of noise / drop rate
        """
        super(GaussDropout, self).__init__()

        self.input_size = input_size

        self.var_init = p
        self.max_alpha = 1.0    # set threshold for alpha
        self.clip = clip

        self.prior_mean = torch.ones(1, self.input_size)
        self.prior_var = (torch.ones(1, self.input_size) * self.var_init).log()
 
        self.muy = nn.Parameter(torch.ones(1, self.input_size))
        self.log_alpha = nn.Parameter((torch.ones(1, self.input_size) * math.sqrt(self.var_init)).log())


    def kld(self):
        log_alpha = self.log_alpha*2

        trace_term = torch.exp(log_alpha - self.prior_var)
        mean_term =  (self.muy - self.prior_mean)**2 * torch.exp(-self.prior_var)
        det_term = self.prior_var - log_alpha

        return 0.5 * torch.sum(trace_term  + det_term + mean_term - 1)


    def forward(self, x, noise= None):
        kld = 0
        if self.training:
            # N(0,1)
            epsilon = torch.randn(x.size()) 
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            if self.clip:
                self.log_alpha.data = torch.clamp(self.log_alpha.data, max= math.log(self.max_alpha - 1e-6))
            alpha = torch.exp(self.log_alpha)
            muy = self.muy * torch.ones(x.size())


            # input vectors have the same alpha vector
            # each feature in each input vector has particular alpha_{i} 
            epsilon = epsilon * alpha + muy
            kld = self.kld()
            
            return x * epsilon, kld

        else:
            s = 1
            if noise != None:
                log_alpha = torch.Tensor(noise[1])
                epsilon = torch.randn(x.size())    
                if x.is_cuda:
                    epsilon = epsilon.cuda()
                
                if self.clip:
                    log_alpha = torch.clamp(log_alpha, max= math.log(self.max_alpha - 1e-6))
                alpha = torch.exp(log_alpha)
                muy = torch.Tensor(noise[0]) * torch.ones(x.size())
                # alpha = math.exp(noise)
                s = epsilon * alpha + muy
            # No scaling 
            return x * s, kld


class GaussDropoutConv2d(nn.Module):
    def __init__(self, in_channels, p=0.5, size=None, clip=True):
        """
        Variational Dropout for Conv2D
        :param in_channels: the number of input's channels
        :param p: initial dropout rate / variance of noise
        :param size: width and height of input
        """

        super(GaussDropoutConv2d, self).__init__()
        self.in_channels = in_channels
        self.size = size
        self.var_init = p
        self.max_alpha = 1.0    # set threshold for alpha
        self.clip = clip

        self.prior_mean = torch.ones(1, in_channels, self.size, self.size)
        self.prior_var = (torch.ones(1, in_channels, self.size, self.size) * self.var_init).log()
 
        self.muy = Parameter(torch.ones(1, in_channels, self.size, self.size))
        self.log_alpha = Parameter((torch.ones(1, in_channels, self.size, self.size) * math.sqrt(self.var_init)).log())

        
    def kld(self):
        log_alpha = self.log_alpha*2

        trace_term = torch.exp(log_alpha - self.prior_var)
        mean_term =  (self.muy - self.prior_mean)**2 * torch.exp(-self.prior_var)
        det_term = self.prior_var - log_alpha

        return 0.5 * torch.sum(trace_term  + det_term + mean_term - 1)


    def forward(self, x, noise= None, clip = True):
        kld = 0
        if self.training:
            # N(0,1)
            epsilon = torch.randn(x.size()) 
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            if self.clip:
                self.log_alpha.data = torch.clamp(self.log_alpha.data, max= math.log(self.max_alpha - 1e-6))
            alpha = torch.exp(self.log_alpha)
            muy = self.muy * torch.ones(x.size())


            # input vectors have the same alpha vector
            # each feature in each input vector has particular alpha_{i} 
            epsilon = epsilon * alpha + muy
            kld = self.kld()
            
            return x * epsilon, kld

        else:
            s = 1
            if noise != None:
                log_alpha = torch.Tensor(noise[1])
                epsilon = torch.randn(x.size())    
                if x.is_cuda:
                    epsilon = epsilon.cuda()
                
                if self.clip:
                    log_alpha = torch.clamp(log_alpha, max= math.log(self.max_alpha - 1e-6))
                alpha = torch.exp(log_alpha)
                muy = torch.Tensor(noise[0]) * torch.ones(x.size())
                # alpha = math.exp(noise)
                s = epsilon * alpha + muy
            # No scaling 
            return x * s, kld