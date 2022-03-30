from networks.gvcl_model_classes import MultiHeadFiLMCNN, MultiHeadFiLMCNNVD
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

from arguments import get_args

args = get_args()

class MLPFilmVD:
    class Net(MultiHeadFiLMCNNVD):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            if not args.single_head:
                super().__init__((1,28,28), [], [256,256], heads, [28*28, 256], film_type = 'point')
            else:
                super().__init__((1,28,28), [], [400,400], heads, [28*28, 400, 400], film_type = 'point')
        
        def forward_linear(self, x, task_labels, num_samples=1, tasks = None):
            for i, layer in enumerate(self.fc_layers):
                x = self.fc_dropout_layers[i](x, task_labels, num_samples)
                x = layer(x)
                if args.film:
                    x = self.fc_film_layers[i](x, task_labels, num_samples)
                x = F.relu(x)
            if args.single_head:
                x = self.fc_dropout_layers[-1](x, task_labels, num_samples)
            return x

        def forward_conv(self, x, task_labels, num_samples=1, tasks = None):
            return x


class CNNFilmVD:
    class Net(MultiHeadFiLMCNNVD):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32, 3, 1), (32, 3, 1), 'pool', (64, 3, 1), (64, 3, 1), 'pool', (128, 3, 1), (128, 3, 1), 'pool'], [256], heads, 
                            [256], film_type = 'point')
            if not args.conv_Dropout:     
                self.drop = nn.Dropout(0.25)

        def forward_conv(self, x, task_labels, num_samples=1, tasks = None):
            drop_index = 0
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if args.film:
                    x = self.conv_film_layers[i](x, task_labels, num_samples)
                
                x = F.relu(x)
                if i in self.pool_indices:
                    x = F.max_pool2d(x, kernel_size = 2, stride = 2)
                    if args.conv_Dropout:
                        x = self.conv_dropout_layers[drop_index](x, task_labels, num_samples)
                    else:
                        x = self.drop(x)
                    drop_index += 1
            return x

        def forward_linear(self, x, task_labels, num_samples=1, tasks = None):
            for i, layer in enumerate(self.fc_layers):
                x = layer(x)
                if args.film:
                    x = self.fc_film_layers[i](x, task_labels, num_samples)
                x = F.relu(x)
                x = self.fc_dropout_layers[i](x, task_labels, num_samples)
            return x

class CNNOmniglotFilmVD:
    class Net(MultiHeadFiLMCNNVD):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [(64, 3, 0), (64, 3, 0), 'pool', (64, 3, 0), (64, 3, 0), 'pool'], [], heads, 
                                        [1024], film_type = 'point')
            if not args.conv_Dropout:     
                self.drop = nn.Dropout(args.droprate_linear)

        def forward_conv(self, x, task_labels, num_samples=1, tasks = None):
            drop_index = 0
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if args.film:
                    x = self.conv_film_layers[i](x, task_labels, num_samples)
                
                x = F.relu(x)
                if i in self.pool_indices:
                    x = F.max_pool2d(x, kernel_size = 2, stride = 2)
                    if args.conv_Dropout:
                        x = self.conv_dropout_layers[drop_index](x, task_labels, num_samples)
                    else:
                        x = self.drop(x)
                    drop_index += 1
            return x

        def forward_linear(self, x, task_labels, num_samples=1, tasks = None):
            if not args.conv_Dropout:
                x = self.fc_dropout_layers[0](x, task_labels, num_samples)
            return x

class MLPFilm:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            if not args.single_head:
                super().__init__((1,28,28), [], [256,256], heads, film_type = 'point')
            else:
                super().__init__((1,28,28), [], [400,400], heads, film_type = 'point')
            if not args.wo_Dropout:
                self.drop = torch.nn.Dropout(args.droprate_linear)
        
        def forward_linear(self, x, task_labels, num_samples=1, tasks = None):
            for i, layer in enumerate(self.fc_layers):
                x = layer(x)
                if args.film:
                    x = self.fc_film_layers[i](x, task_labels, num_samples)
                x = F.relu(x)
                if not args.wo_Dropout:
                    x = self.drop(x)
            return x

        def forward_conv(self, x, task_labels, num_samples=1, tasks = None):
            return x

class CNNOmniglotFilm:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [(64, 3, 0), (64, 3, 0), 'pool', (64, 3, 0), (64, 3, 0), 'pool'], [], heads, 
                                        film_type = 'point')
            if not args.wo_Dropout:
                self.drop = nn.Dropout(args.droprate_linear)

        def forward_conv(self, x, task_labels, num_samples=1, tasks = None):
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if args.film:
                    x = self.conv_film_layers[i](x, task_labels, num_samples)
                
                x = F.relu(x)
                if i in self.pool_indices:
                    x = F.max_pool2d(x, kernel_size = 2, stride = 2)
                    if not args.wo_Dropout:
                        x = self.drop(x)
            return x

        def forward_linear(self, x, task_labels, num_samples=1, tasks = None):
            return x

class CNNFilm:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32, 3, 1), (32, 3, 1), 'pool', (64, 3, 1), (64, 3, 1), 'pool', (128, 3, 1), (128, 3, 1), 'pool'], [256], heads, 
                                        film_type = 'point')
            if not args.wo_Dropout:
                self.drop1 = nn.Dropout(0.25)
                self.drop2 = nn.Dropout(0.5)

        def forward_conv(self, x, task_labels, num_samples=1, tasks = None):
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if args.film:
                    x = self.conv_film_layers[i](x, task_labels, num_samples)
                
                x = F.relu(x)
                if i in self.pool_indices:
                    x = F.max_pool2d(x, kernel_size = 2, stride = 2)
                    if not args.wo_Dropout:
                        x = self.drop1(x)
            return x

        def forward_linear(self, x, task_labels, num_samples=1, tasks = None):
            for i, layer in enumerate(self.fc_layers):
                x = layer(x)
                if args.film:
                    x = self.fc_film_layers[i](x, task_labels, num_samples)
                x = F.relu(x)
                if not args.wo_Dropout:
                    x = self.drop2(x)
            return x



