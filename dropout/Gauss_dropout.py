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

device = 'cuda:0'

class GaussDropout(nn.Module):
    def __init__(self, tasks, input_size, p=0.5):
        super(GaussDropout, self).__init__()

        self.input_size = input_size
        self.muy = nn.Parameter(torch.Tensor(tasks, self.input_size))

        self.var_init = p
        self.prior_mean = torch.ones([1, input_size], device = device)
        self.prior_var = torch.ones([1, input_size], device = device).mul(np.log(self.var_init))

        self.p = p      
        self.log_alpha = nn.Parameter(torch.Tensor(tasks, self.input_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.muy, 1)

        alpha = math.sqrt(self.var_init)
        init.constant_(self.log_alpha, math.log(alpha))

    def forward(self, x, task_labels, num_samples=1):
        # # x.shape = [num_samples, batch_size, input_size]
        muy = F.embedding(task_labels, self.muy)
        muy.view(1, -1, self.input_size)
        muy = muy * torch.ones(x.size()).cuda() 

        log_alpha = F.embedding(task_labels, self.log_alpha)
        epsilon = torch.randn(x.size()) 
        if x.is_cuda:
            epsilon = epsilon.cuda()
        
        alpha = torch.exp(log_alpha)
        alpha.view(1, -1, self.input_size)

        # input vectors have the same alpha vector
        # each feature in each input vector has particular alpha_{i} 
        epsilon = epsilon * alpha + muy

        return x * epsilon  

    def get_kl(self, task, sum=True):

        task = torch.tensor([task]).cuda()
        log_alpha = F.embedding(task, self.log_alpha)*2
        muy = F.embedding(task, self.muy)


        trace_term = torch.exp(log_alpha - self.prior_var)
        mean_term =  (muy - self.prior_mean)**2 * torch.exp(-self.prior_var)
        det_term = self.prior_var - log_alpha

        if sum:
            return 0.5 * torch.sum(trace_term  + det_term + mean_term - 1)
        else:
            return 0.5 * (trace_term + mean_term + det_term - 1) 

 

class GaussDropoutConv2d(nn.Module):
    def __init__(self, tasks, in_channels, size=None, p=0.5):
        super(GaussDropoutConv2d, self).__init__()

        # Initial alpha
        self.p = p      
        self.in_channels = in_channels
        self.size = size
        self.log_alpha = nn.Parameter(torch.Tensor(tasks, in_channels * self.size * self.size))
        
        self.muy = nn.Parameter(torch.Tensor(tasks, in_channels * self.size * self.size))
        self.var_init = p
        self.prior_mean = torch.ones([1, in_channels * self.size * self.size], device = device)
        self.prior_var = torch.ones([1, in_channels * self.size * self.size], device = device).mul(np.log(self.var_init))

        self.reset_parameters()
    
    def reset_parameters(self):
        init.constant_(self.muy, 1)

        alpha = math.sqrt(self.var_init)
        init.constant_(self.log_alpha, math.log(alpha))

    def forward(self, x, task_labels, num_samples=1):
        # x.shape = [batch_size, channel, h, w]
        muy = F.embedding(task_labels, self.muy)
        muy = muy.reshape([x.shape[0]//num_samples, self.in_channels, self.size, self.size]).repeat(num_samples, 1,1,1)

        #self.log_alpha.data = torch.clamp(self.log_alpha.data, max= math.log(self.max_alpha - 1e-6))
        log_alpha = F.embedding(task_labels, self.log_alpha)
        epsilon = torch.randn(x.size()) 
        if x.is_cuda:
            epsilon = epsilon.cuda()
        
        alpha = torch.exp(log_alpha)
        alpha = alpha.reshape([x.shape[0]//num_samples, self.in_channels, self.size, self.size]).repeat(num_samples, 1,1,1)

        epsilon = epsilon * alpha + muy
        
        return x * epsilon  
    


    def get_kl(self, task, sum=True):
        task = torch.tensor([task]).cuda()
        log_alpha = F.embedding(task, self.log_alpha)*2
        muy = F.embedding(task, self.muy)


        trace_term = torch.exp(log_alpha - self.prior_var)
        mean_term =  (muy - self.prior_mean)**2 * torch.exp(-self.prior_var)
        det_term = self.prior_var - log_alpha

        if sum:
            return 0.5 * torch.sum(trace_term  + det_term + mean_term - 1)
        else:
            return 0.5 * (trace_term + mean_term + det_term - 1) 