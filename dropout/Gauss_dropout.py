import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import math
from torch.nn import init
from functools import partial
from torch.nn.modules.utils import _pair

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import distributions

import math
import numpy as np
from torch.nn import init
from functools import partial

class GaussDropout(nn.Module):
    def __init__(self, tasks, input_size, p=0.5):
        """
            Variational Dropout
            :param input_size: An int of input size
            :param p: An initial variance of noise / drop rate
        """
        super(GaussDropout, self).__init__()

        # Initial alpha
        self.p = p      
        self.input_size = input_size
        self.max_alpha = 1.0
        self.log_alpha = nn.Parameter(torch.Tensor(tasks, self.input_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        alpha = math.sqrt(self.p/(1-self.p))
        init.constant_(self.log_alpha, math.log(alpha))

    def forward(self, x, task_labels, num_samples=1):
        # x.shape = [num_samples, batch_size, input_size]
        self.log_alpha.data = torch.clamp(self.log_alpha.data, max= math.log(self.max_alpha - 1e-6))
        log_alpha = F.embedding(task_labels, self.log_alpha)
        epsilon = torch.randn(x.size()) 
        if x.is_cuda:
            epsilon = epsilon.cuda()
        
        #log_alpha.data = torch.clamp(log_alpha.data, max= math.log(self.max_alpha - 1e-6))
        alpha = torch.exp(log_alpha)
        alpha.view(1, -1, self.input_size)

        # input vectors have the same alpha vector
        # each feature in each input vector has particular alpha_{i} 
        epsilon = epsilon * alpha + 1

        
        return x * epsilon  

    def get_kl(self, task):
        """
            Calculate KL-divergence between N(1, alpha) and log-uniform prior
            This approximated KL is calculated follow the Kingma's paper
            https://arxiv.org/abs/1506.02557
        """
        task = torch.tensor([task]).cuda()
        log_alpha = F.embedding(task, self.log_alpha)
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        alpha = (2 * log_alpha).exp()         # self.log_alpha was clipped to ensure less equal than zero   
        negative_kl = log_alpha + c1*alpha + c2*alpha**2 + c3*alpha**3          
        kl = -negative_kl

        
        return kl.sum()   
 

class GaussDropoutConv2d(nn.Module):
    def __init__(self, tasks, in_channels, size=None, p=0.5):
        """
            Variational Dropout
            :param input_size: An int of input size
            :param p: An initial variance of noise / drop rate
        """
        super(GaussDropoutConv2d, self).__init__()

        # Initial alpha
        self.p = p      
        self.in_channels = in_channels
        self.size = size
        self.max_alpha = 1.0
        self.log_alpha = nn.Parameter(torch.Tensor(tasks, in_channels * self.size * self.size))
        self.reset_parameters()
    
    def reset_parameters(self):
        alpha = math.sqrt(self.p/(1-self.p))
        init.constant_(self.log_alpha, math.log(alpha))

    def forward(self, x, task_labels, num_samples=1):
        # x.shape = [batch_size, channel, h, w]
        self.log_alpha.data = torch.clamp(self.log_alpha.data, max= math.log(self.max_alpha - 1e-6))
        log_alpha = F.embedding(task_labels, self.log_alpha)
        epsilon = torch.randn(x.size()) 
        if x.is_cuda:
            epsilon = epsilon.cuda()
        
        # log_alpha.data = torch.clamp(log_alpha.data, max= math.log(self.max_alpha - 1e-6))
        alpha = torch.exp(log_alpha)
        alpha = alpha.reshape([x.shape[0]//num_samples, self.in_channels, self.size, self.size]).repeat(num_samples, 1,1,1)


        # input vectors have the same alpha vector
        # each feature in each input vector has particular alpha_{i} 
        epsilon = epsilon * alpha + 1
        
        return x * epsilon  
    


    def get_kl(self, task):
        """
            Calculate KL-divergence between N(1, alpha) and log-uniform prior
            This approximated KL is calculated follow the Kingma's paper
            https://arxiv.org/abs/1506.02557
        """
        task = torch.tensor([task]).cuda()
        log_alpha = F.embedding(task, self.log_alpha)
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        alpha = (2 * log_alpha).exp()         # self.log_alpha was clipped to ensure less equal than zero   
        negative_kl = log_alpha + c1*alpha + c2*alpha**2 + c3*alpha**3          
        kl = -negative_kl

        
        return kl.sum() 