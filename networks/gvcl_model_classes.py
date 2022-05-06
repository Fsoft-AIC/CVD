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

from abc import ABC, abstractmethod
from utils import *
from arguments import get_args


device = 'cuda:0'

args = get_args()

if args.drop_prior == 'gauss':
    from dropout.Gauss_dropout import GaussDropoutConv2d 
    from dropout.Gauss_dropout import GaussDropout 
elif args.drop_prior == 'log_uniform':
    from dropout.LUGauss_dropout import GaussDropoutConv2d 
    from dropout.LUGauss_dropout import GaussDropout 

class MultiHeadFiLMCNN(nn.Module):
    def __init__(self, input_shape, conv_sizes, fc_sizes, output_dims, film_type = 'point', global_avg_pool = False, prior_var = -1, init_vars = []):
        super(MultiHeadFiLMCNN, self).__init__()
        self.conv_layers = nn.ModuleList([])
        self.fc_layers = nn.ModuleList([])
        self.heads = nn.ModuleList([])
        self.num_tasks = len(output_dims)
        self.output_dims = output_dims
        self.single_head = args.single_head
        self.prior_var = args.prior_var
        self.global_avg_pool = global_avg_pool
        self.pool_indices = []        
        
        if args.film:
            self.film_type = args.film_type
            self.set_film_gen_type()
            print("Film type:", self.film_type)
            self.conv_film_layers = nn.ModuleList([self.conv_film_gen_type(self.num_tasks, conv_size[0]) for conv_size in conv_sizes if conv_size != 'pool'])
            self.fc_film_layers = nn.ModuleList([self.fc_film_gen_type(self.num_tasks, fc_size) for fc_size in fc_sizes])


        self.prior_var = args.prior_var

        if len(init_vars) == 0:
            init_vars = -7 * np.ones([len(conv_sizes) + len(fc_sizes) + 1])
        
        layer_index = 0

        channels = input_shape[0]
        prev_channels = input_shape[0]
        input_dimension = input_shape[1]
        for i, layer_params in enumerate(conv_sizes):
            if layer_params == 'pool':
                self.pool_indices.append(i - len(self.pool_indices) - 1)
                input_dimension = input_dimension//2
                continue
            channels = layer_params[0]
            kernel_size = layer_params[1]
            if(len(layer_params) == 2):
                padding = int((kernel_size-1)/2)
            else:
                padding = layer_params[2]
            conv_layer = MFConvLayer(prev_channels, channels, kernel_size=kernel_size, stride=1, padding=padding, prior_var = self.prior_var, init_var = init_vars[layer_index])
            
            input_dimension = int(np.floor((input_dimension+2*padding-(kernel_size-1)-1)/float(1)+1))
            prev_channels = channels
            self.conv_layers.append(conv_layer)
            layer_index += 1
        
        print(layer_index)

        last_size = channels * input_dimension**2
        
        if global_avg_pool:
            last_size = channels
        
        for i, hidden_size in enumerate(fc_sizes):
            self.fc_layers.append(MFLinearLayer(last_size, hidden_size, prior_var = self.prior_var, init_var = init_vars[layer_index]))
            last_size = hidden_size
            layer_index += 1

        last_size_copy = last_size
        for task in range(self.num_tasks):
            last_size_copy = last_size
            layers = nn.ModuleList([])
        last_size = last_size_copy
        
        if self.single_head:
            self.heads.append(MFLinearLayer(last_size, output_dims[0], prior_var = self.prior_var, init_var = init_vars[layer_index]))
        else:
            for output_dim in output_dims:
                self.heads.append(MFLinearLayer(last_size, output_dim, prior_var = self.prior_var, init_var = init_vars[layer_index]))

    def get_task_specific_parameters(self, task_number):
        modules = nn.ModuleList()
        if args.film:
            modules.append(self.conv_film_layers)
            modules.append(self.fc_film_layers)
        if not self.single_head:
            modules.append(self.heads[task_number])
        modules.append(self.fc_layers)
        modules.append(self.conv_layers)
        if self.single_head:
            modules.append(self.heads[0])
        
        return modules.parameters()

    def set_film_gen_type(self):
        if(self.film_type == 'point'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = False, conv = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = False)
        elif(self.film_type == 'scale'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = False, conv = True, scale_only = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = False, scale_only = True)
        elif(self.film_type == 'bias'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = False, conv = True, bias_only = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = False, bias_only = True)
        elif(self.film_type == 'none'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = True, conv = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = True)

    def to(self, device):
        self.device = device
        return super().to(device)


    def forward(self, x, task_labels, num_samples=1, tasks = None):
        if tasks is None:
            tasks = range(self.num_tasks)
            excluded_tasks = []
        else:
            excluded_tasks = [i for i in range(self.num_tasks) if i not in tasks]

        num_total_tasks = len(tasks)

        outputs = [None for j in range(self.num_tasks)]

        batch_size = x.shape[0]

        x = x.repeat([num_samples,1,1,1])

        x = self.forward_conv(x, task_labels, num_samples, tasks)
        
        if self.global_avg_pool:
            x = x.view(num_samples, batch_size, x.shape[1], -1).mean(-1)
        else:
            x = x.view(num_samples, batch_size, -1)

        x = self.forward_linear(x, task_labels, num_samples, tasks)
            
        self.pre_head = x

        for j in tasks:
            head_index = 0 if self.single_head else j
            task_output = self.heads[head_index](x)
            outputs[j] = task_output.reshape([num_samples, batch_size, -1])
        for j in excluded_tasks:
            outputs[j] = torch.zeros_like(task_output, device = device)

        return outputs


    def get_kl(self, lamb = 1):
        kl = 0

        for i, conv_layer in enumerate(self.conv_layers):
            kl += conv_layer.get_kl(lamb)
        
        for layer in self.fc_layers:
            kl += layer.get_kl(lamb)

        for t, layer in enumerate(self.heads):
            kl += layer.get_kl(lamb)

        return kl

    
    def add_task_body_params(self, updated_tasks):
        for layer in self.fc_layers:
            layer.add_new_task()
        for layer in self.conv_layers:
            layer.add_new_task()
        if self.single_head:
            self.heads[0].add_new_task(reset_variance = False)
        if not self.single_head:
            for t in updated_tasks:
                self.heads[t].add_new_task(reset_variance = False)

class MultiHeadFiLMCNNVD(MultiHeadFiLMCNN):
    def __init__(self, input_shape, conv_sizes, fc_sizes, output_dims, drop_fc_sizes, film_type = 'point', global_avg_pool = False, prior_var = -1, init_vars = []):
        super().__init__(input_shape, conv_sizes, fc_sizes, output_dims, film_type, global_avg_pool, prior_var, init_vars)

        self.set_dropout_gen_type()
        print("Drop prior:", args.drop_prior)
        if args.conv_Dropout:
            self.conv_dropout_layers = nn.ModuleList()
            s = input_shape[-1]
            prev_channel = input_shape[0]
            for i, layer_params in enumerate(conv_sizes):
                if layer_params != 'pool':
                    #self.conv_dropout_layers.append(self.conv_dropout_gen_type(self.num_tasks, prev_channel, s))
                    s = compute_conv_output_size(s, kernel_size=layer_params[1], padding=layer_params[2])
                    prev_channel = layer_params[0]
                    self.conv_dropout_layers.append(self.conv_dropout_gen_type(self.num_tasks, prev_channel, s))

                else:
                    s = s//2
                    #self.conv_dropout_layers.append(self.conv_dropout_gen_type(self.num_tasks, prev_channel, s))
        self.fc_dropout_layers = nn.ModuleList([self.fc_dropout_gen_type(self.num_tasks, drop_fc_size) for drop_fc_size in drop_fc_sizes])

    def set_dropout_gen_type(self):
        if args.conv_Dropout:
            self.conv_dropout_gen_type = partial(GaussDropoutConv2d, p = args.droprate)
        self.fc_dropout_gen_type = partial(GaussDropout, p = args.droprate)

    def get_task_specific_parameters(self, task_number):
        modules = nn.ModuleList([self.fc_dropout_layers])
        if args.conv_Dropout:
            modules.append(self.conv_dropout_layers)
        if args.film:
            modules.append(self.conv_film_layers)
            modules.append(self.fc_film_layers)
        if not self.single_head:
            modules.append(self.heads[task_number])
        modules.append(self.fc_layers)
        modules.append(self.conv_layers)
        if self.single_head:
            modules.append(self.heads[0])
        
        return modules.parameters()

    def get_dropout_kl(self, task):
        kl = 0

        if args.conv_Dropout:
            for layer in self.conv_dropout_layers:
                kl += layer.get_kl(task)

        for layer in self.fc_dropout_layers:
            kl += layer.get_kl(task)
        
        return kl 


class PointFiLMLayer(nn.Module):
    def __init__(self, tasks, width, constant = False, conv = False, scale_only = False, bias_only = False):
        super().__init__()
        self.scales = Parameter(torch.Tensor(tasks, width), requires_grad = (not constant) and (not bias_only))
        self.shifts = Parameter(torch.Tensor(tasks, width), requires_grad = (not constant) and (not scale_only))
        self.conv = conv
        self.width = width
        self.constant = constant
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.scales, 1.)
        init.constant_(self.shifts, 0.)
    
    def forward(self, x, task_labels, num_samples):
        scale_values = F.embedding(task_labels, self.scales)
        shift_values = F.embedding(task_labels, self.shifts)

        if self.conv:
            scale_values = scale_values.view(-1, self.width, 1, 1).repeat(num_samples, 1,1,1)
            shift_values = shift_values.view(-1, self.width, 1, 1).repeat(num_samples, 1,1,1)
        else:
            scale_values = scale_values.view(1, -1, self.width)
            shift_values = shift_values.view(1, -1, self.width)

        return x * scale_values + shift_values

class MFConvLayer(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', prior_var = 1, init_var = -7, ratio=0.5):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.init_var = init_var
        
        
        self.W_prior_mean = torch.zeros(self.weight.shape, device = device)
        self.b_prior_mean = torch.zeros(self.bias.shape, device = device)


        if prior_var == -1:
            _, fan_out = _calculate_fan_in_and_fan_out(self.weight)
            gain = 1 # Var[w] + sigma^2 = 2/fan_in
            
            total_var = 2 / fan_out
            noise_var = total_var * ratio
            mu_var = total_var - noise_var
            
            noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
            bound = math.sqrt(3.0) * mu_std
            rho_init = np.log(np.exp(noise_std)-1)
            bias_rho_init = np.log(np.exp(1) - 1)

            self.w_var_init = np.log1p(np.exp(rho_init))**2
            self.b_var_init = np.log1p(np.exp(bias_rho_init))**2
        else :
            self.w_var_init = prior_var
            self.b_var_init = prior_var

        
        self.W_prior_var = torch.ones(self.weight.shape, device = device).mul(np.log(self.w_var_init))
        self.b_prior_var = torch.ones(self.bias.shape, device = device).mul(np.log(self.b_var_init))
        self.weight_var = Parameter(torch.Tensor(self.weight.shape))
        self.bias_var = Parameter(torch.Tensor(self.bias.shape))
        
        self.reset_parameters()

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'weight_var'):
            init.constant_(self.weight_var, self.init_var)
            init.constant_(self.bias_var, self.init_var)

    def add_new_task(self):
        self.W_prior_mean = self.weight.clone().detach().requires_grad_(False)
        self.b_prior_mean = self.bias.clone().detach().requires_grad_(False)
        
        self.W_prior_var = self.weight_var.clone().detach().requires_grad_(False)
        self.b_prior_var = self.bias_var.clone().detach().requires_grad_(False)
        
        self.weight_var.data = torch.min(self.weight_var, self.init_var*torch.ones_like(self.weight_var).data)
        self.bias_var.data = torch.min(self.bias_var, self.init_var*torch.ones_like(self.bias_var).data)

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)

        initialization_noise = torch.empty_like(self.weight)
        init.kaiming_uniform_(initialization_noise, a = math.sqrt(5))
        # self.weight.data = self.weight.data + (self.weight_var > -2).float() * initialization_noise
        # self.bias.data = self.bias.data + (self.bias_var > -2).float() * torch.empty_like(self.bias).uniform_(-bound, bound)

        self.weight.data = initialization_noise.data
        self.bias.data = torch.empty_like(self.bias).uniform_(-bound, bound).data

    def get_kl(self, lamb):
        W_kl = compute_kl(self.weight, self.weight_var, self.W_prior_mean, self.W_prior_var, lamb = lamb, initial_prior_var = self.w_var_init)
        b_kl = compute_kl(self.bias, self.bias_var, self.b_prior_mean, self.b_prior_var, lamb = lamb, initial_prior_var = self.b_var_init)

        return W_kl + b_kl

    def forward(self, input):
        output_mean =  self.conv2d_forward(input, self.weight, self.bias)
        output_var = self.conv2d_forward(input**2, torch.exp(self.weight_var), torch.exp(self.bias_var))

        eps = torch.empty(output_mean.shape, device=device).normal_(mean=0,std=1)
        output = output_mean + torch.sqrt(output_var + 1e-9) * eps

        return output 


class MFLinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, prior_var = -1, init_var = -7, ratio=0.5):
        super().__init__()
        self.init_var = init_var
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W_mean = Parameter(torch.Tensor(dim_out, dim_in))
        self.b_mean = Parameter(torch.Tensor(dim_out))

        self.W_var = Parameter(torch.Tensor(dim_out, dim_in))
        self.b_var = Parameter(torch.Tensor(dim_out))

        self.W_prior_mean = torch.zeros([dim_out, dim_in], device = device)
        self.b_prior_mean = torch.zeros([dim_out], device = device)


        if prior_var == -1:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.W_mean)
            gain = 1 # Var[w] + sigma^2 = 2/fan_in
            
            total_var = 2 / fan_in
            noise_var = total_var * ratio
            mu_var = total_var - noise_var
            
            noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
            bound = math.sqrt(3.0) * mu_std
            rho_init = np.log(np.exp(noise_std)-1)
            bias_rho_init = np.log(np.exp(1) - 1)

            self.w_var_init = np.log1p(np.exp(rho_init))**2
            self.b_var_init = np.log1p(np.exp(bias_rho_init))**2
        else :
            self.w_var_init = prior_var
            self.b_var_init = prior_var
        
        self.W_prior_var = torch.ones([dim_out, dim_in], device = device).mul(np.log(self.w_var_init))
        self.b_prior_var = torch.ones([dim_out], device = device).mul(np.log(self.b_var_init))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_mean, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_mean)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b_mean, -bound, bound)

        init.constant_(self.W_var, self.init_var)
        init.constant_(self.b_var, self.init_var)

    def add_new_task(self, reset_variance = True):
        self.W_prior_mean = self.W_mean.clone().detach().requires_grad_(False)
        self.b_prior_mean = self.b_mean.clone().detach().requires_grad_(False)
    
        self.W_prior_var = self.W_var.clone().detach().requires_grad_(False)
        self.b_prior_var = self.b_var.clone().detach().requires_grad_(False)
        
        if reset_variance:
            self.W_var.data = torch.min(self.W_var, self.init_var*torch.ones_like(self.W_var).data)
            self.b_var.data = torch.min(self.b_var, self.init_var*torch.ones_like(self.b_var).data)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_mean)
            bound = 1 / math.sqrt(fan_in)

            initialization_noise = torch.empty_like(self.W_mean)
            init.kaiming_uniform_(initialization_noise, a = math.sqrt(5))
            # self.W_mean.data = self.W_mean.data + (self.W_var > -2).float() * initialization_noise
            # self.b_mean.data = self.b_mean.data + (self.b_var > -2).float() * torch.empty_like(self.b_mean).uniform_(-bound, bound)

            self.W_mean.data = initialization_noise.data
            self.b_mean.data = torch.empty_like(self.b_mean).uniform_(-bound, bound).data

    def get_kl(self, lamb):
        W_kl = compute_kl(self.W_mean, self.W_var, self.W_prior_mean, self.W_prior_var, lamb = lamb, initial_prior_var = self.w_var_init)
        b_kl = compute_kl(self.b_mean, self.b_var, self.b_prior_mean, self.b_prior_var, lamb = lamb, initial_prior_var = self.b_var_init)
        return W_kl + b_kl

    def forward(self, x):
        output_mean = x.matmul(self.W_mean.t()) + self.b_mean.unsqueeze(0).unsqueeze(0)
        output_std = torch.sqrt((x**2).matmul(torch.exp(self.W_var.t())) + torch.exp(self.b_var).unsqueeze(0).unsqueeze(0))
        eps = torch.empty(output_mean.shape, device=device).normal_(mean=0,std=1)

        output = output_mean + (eps * output_std)
        return output



def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out