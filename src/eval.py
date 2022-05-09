import torch
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

import utils
from utils import *
from arguments import get_args

args = get_args()



lt=0
if args.local_trick:
    lt = 1

if args.approach == 'si':
    log_name = '{}_{}_{}_c_{}_lr_{}_unitN_{}_batch_{}_epoch_{}'.format(args.experiment, args.approach,args.seed, 
                                                                    args.c, args.lr, args.unitN, args.batch_size, args.nepochs)
elif args.approach == 'ewc' or args.approach == 'rwalk' or args.approach == 'mas':
    log_name = '{}_{}_{}_woDr_{}_lamb_{}_lr_{}_unitN_{}_batch_{}_epoch_{}'.format(args.experiment, args.approach,args.seed,
                                                                                args.wo_Dropout, args.lamb, args.lr, args.unitN, 
                                                                                args.batch_size, args.nepochs)                                                                         
elif args.approach == 'ewc_vd' or args.approach == 'si_vd' or args.approach == 'mas_vd':
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_lamb_{}_lr_{}_unitN_{}_batch_{}_epoch_{}'.format(
                                                                args.experiment, args.approach,args.seed,
                                                                args.KL_coeff, args.KL_weight, args.num_samples,
                                                                args.conv_Dropout, args.drop_type, args.droprate,
                                                                args.lamb, args.lr, args.unitN, 
                                                                args.batch_size, args.nepochs)
if args.approach == 'vcl':
    log_name = '{}_{}_{}_woDr_{}_lr_{}_KLtheta_{}_local-trick_{}_unitN_{}_batch_{}_epoch_{}'.format(args.experiment, args.approach,args.seed, 
                                                                     args.wo_Dropout, args.lr, args.KL_weight_theta, lt, args.unitN, args.batch_size, args.nepochs)


elif args.approach == 'vcl_vd':
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_KLtheta_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_lr_{}_local-trick_{}_unitN_{}_batch_{}_epoch_{}'.format(
                                                                args.experiment, args.approach,args.seed,
                                                                args.KL_coeff, args.KL_weight, args.KL_weight_theta, 
                                                                args.num_samples, args.conv_Dropout, args.drop_type, args.droprate,
                                                                args.lr, lt, args.unitN, 
                                                                args.batch_size, args.nepochs)
elif args.approach == 'ucl_vd':
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_lr_{}_local-trick_{}_lr_rho_{}_epoch_{}'.format(
        args.experiment, args.approach, args.seed,  
        args.KL_coeff, args.KL_weight, args.num_samples, 
        args.conv_Dropout, args.drop_type, args.droprate, 
        args.alpha, args.beta, args.ratio, args.lr, lt, 
        args.lr_rho, args.nepochs)


elif args.approach == 'ucl' or args.approach == 'baye_hat':
    log_name = '{}_{}_{}_wo_Dropout_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_lr_{}_local-trick_{}_lr_rho_{}_unitN_{}_batch_{}_epoch_{}'.format(
        args.experiment, args.approach, args.seed,  args.wo_Dropout, args.alpha, args.beta, args.ratio, 
        args.lr, lt, args.lr_rho, args.unitN, args.batch_size, args.nepochs)

elif args.approach == 'ewc_ablation':
    log_name = '{}_{}_{}_lamb_{}_lr_{}_unitN_{}_batch_{}_epoch_{}'.format(args.experiment, args.approach,args.seed,
                                                                       args.lamb, args.lr, args.unitN, 
                                                                             args.batch_size, args.nepochs)  

elif args.approach == 'ucl_ablation':
    log_name = '{}_{}_{}_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_lr_{}_lr_rho_{}_unitN_{}_batch_{}_epoch_{}'.format(
        args.experiment, args.approach, args.seed, args.ablation, args.alpha, args.beta, args.ratio, 
        args.lr, args.lr_rho, args.unitN, args.batch_size, args.nepochs)

elif args.approach == 'hat':
    log_name = '{}_{}_{}_alpha_{}_smax_{}_lr_{}_unitN_{}_batch_{}_epoch_{}'.format(args.experiment, 
                                                                              args.approach, args.seed,
                                                                              args.alpha, args.smax, args.lr, args.unitN, 
                                                                              args.batch_size, args.nepochs)

########################################################################################################################
# Split
split = False
notMNIST = False
split_experiment = [
    'split_mnist', 
    'split_notmnist', 
    'split_cifar10',
    'split_cifar100',
    'split_cifar100_20',
    'split_cifar10_100',
    'split_pmnist',
    'split_row_pmnist', 
    'split_CUB200',
    'split_tiny_imagenet',
    'split_mini_imagenet', 
    'omniglot',
    'mixture',
    'alter_cifar'
]

conv_experiment = [
    'split_cifar10',
    'split_cifar100',
    'split_cifar100_20',
    'split_cifar10_100',
    'split_CUB200',
    'split_tiny_imagenet',
    'split_mini_imagenet', 
    'omniglot',
    'mixture'
]

if args.experiment in split_experiment:
    split = True
if args.experiment == 'split_notmnist':
    notMNIST = True
if args.experiment in conv_experiment:
    args.conv = True
    log_name = log_name + '_conv'
if args.output == '':
    args.output = './result_data/' + args.experiment + '/' \
                    + args.approach + '/' \
                    + log_name + '.txt'

# Arguments info
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
###############

if not os.path.isdir('./result_data/'+ args.experiment + '/' + args.approach + '/'):
    os.makedirs('./result_data/'+ args.experiment + '/' + args.approach + '/')

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment == 'mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment == 'pmnist' or args.experiment == 'split_pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment == 'row_pmnist' or args.experiment == 'split_row_pmnist':
    from dataloaders import row_pmnist as dataloader
elif args.experiment == 'split_mnist':
    from dataloaders import split_mnist as dataloader
elif args.experiment == 'split_notmnist':
    from dataloaders import split_notmnist as dataloader

elif args.experiment == 'split_cifar100':
    from dataloaders import split_cifar100 as dataloader

elif args.experiment == 'split_cifar10_100':
    from dataloaders import split_cifar10_100 as dataloader

elif args.experiment == 'omniglot':
    from dataloaders import split_omniglot as dataloader
elif args.experiment == 'mixture':
    from dataloaders import mixture as dataloader
elif args.experiment == 'alter_cifar':
    from dataloaders import alter_cifar as dataloader
# Args -- Approach

if args.approach == 'ucl':
    from approaches import ucl as approach
elif args.approach == 'vcl' :
    from approaches import vcl as approach
elif args.approach == 'vcl_snoise' :
    from approaches import vcl_snoise as approach
elif args.approach == 'ewc' or args.approach == 'ewc_ablation':
    from approaches import ewc as approach
elif args.approach == 'si':
    from approaches import si as approach
elif args.approach == 'rwalk':
    from approaches import rwalk as approach
elif args.approach == 'mas':
    from approaches import mas as approach
elif args.approach == 'hat-test':
    from approaches import hat_test as approach
elif args.approach == 'hat':
    from approaches import hat as approach
elif args.approach == 'ewc_vd':
    from approaches import ewc_vd as approach
elif args.approach == 'vcl_vd':
    from approaches import vcl_vd as approach
elif args.approach == 'ucl_vd':
    from approaches import ucl_vd as approach

# Args -- Network

if args.experiment == 'split_cifar100' or args.experiment == 'split_cifar10_100' or args.experiment == 'split_cifar10' or args.experiment == 'split_cifar100_20':
    
    if args.approach == 'hat':
        from networks import conv_net_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import conv_net_ucl as network
    elif args.approach == 'ucl_vd' or args.approach == 'vcl_vd':
        from networks import conv_net_bnn_vd as network       
    elif args.approach == 'ewc_vd' or args.approach == 'si_vd' or args.approach == 'mas_vd':
        from networks import conv_net_ewc_vd as network
    elif args.approach == 'vcl':
        from networks import conv_net_ucl as network  
    elif args.approach == 'ewc_ablation':
        from networks import conv_net_no_dropout as network          
    else:
        from networks import conv_net as network

elif args.experiment == 'mixture' or args.experiment == 'alter_cifar' or args.experiment == 'split_tiny_imagenet' or args.experiment == 'split_CUB200':
    if args.approach == 'hat':
        from networks import alexnet_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import alexnet_ucl as network 
    elif args.approach == 'ewc_vd':
        from networks import alexnet_vd as network
    else:
        from networks import alexnet as network

elif args.experiment == 'omniglot':
    if args.approach == 'hat':
        from networks import conv_net_omniglot_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation' or args.approach == 'vcl':
        from networks import conv_net_omniglot_ucl as network
    elif args.approach == 'ewc_vd':
        from networks import conv_net_omniglot_vd as network
    elif args.approach == 'vcl_vd' or args.approach == 'ucl_vd':
        from networks import conv_net_omniglot_bnn_vd as network
    else:
        from networks import conv_net_omniglot as network
else:
    if args.approach == 'hat':
        from networks import mlp_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import mlp_ucl as network
    elif args.approach == 'ewc_vd' or args.approach == 'si_vd' or args.approach == 'mas_vd':
        from networks import mlp_vd as network
    elif args.approach == 'ucl_vd' or args.approach == 'vcl_vd':
        from networks import mlp_bnn_vd as network
    elif args.approach == 'vcl':
        from networks import mlp_vcl as network
    elif args.approach == 'ewc_ablation':
        from networks import mlp_no_dropout as network        
    else:
        from networks import mlp as network
    

########################################################################################################################

# Load data
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed, tasknum=args.tasknum)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
# print (inputsize,taskcla)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if args.conv_net == False:
    if args.approach == 'ucl' or args.approach == 'ucl_vd' or args.approach == 'vcl' or args.approach == 'vcl_vd':
        net = network.Net(inputsize, taskcla, args.ratio, unitN=args.unitN,  split = split, notMNIST=notMNIST).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
    else:
        net = network.Net(inputsize, taskcla, unitN=args.unitN,  split = split, notMNIST=notMNIST).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
else:
    if args.approach == 'ucl' or args.approach == 'vcl' or args.approach == 'ucl_ablation' or args.approach == 'vcl_vd' or args.approach == 'ucl_vd':
        net = network.Net(inputsize, taskcla, args.ratio).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args,
                             log_name=log_name, split=split)
    else:
        net = network.Net(inputsize, taskcla).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args,
                             log_name=log_name, split=split)

    
num_params = utils.print_model_report(net)

print('-' * 100)

model_file = "model/"+log_name+".pkl"

print("Loading model from " + model_file)
appr = torch.load(model_file)

avg_acc = 0
for u in range(len(taskcla)):
    xtest = data[u]['test']['x'].cuda()
    ytest = data[u]['test']['y'].cuda()
    test_loss, test_acc = appr.eval(u, xtest, ytest)
    avg_acc += test_acc/len(taskcla)
    print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
    # acc[t, u] = test_acc
    # lss[t, u] = test_loss

print("Average acc= {:.2f}%".format(100 * avg_acc))
