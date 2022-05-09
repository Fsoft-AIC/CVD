import sys, os, time
import numpy as np
import torch
import torch.nn as nn

import utils
from utils import *
from arguments import get_args


tstart = time.time()

# Arguments

args = get_args()
# args_std = np.log(1+np.exp(args.rho))

# Using local trick or not (only with BNN)
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
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_lamb_{}_lr_{}_unitN_{}_batch_{}_epoch_{}_clip_{}_drop_prior_{}'.format(
                                                                args.experiment, args.approach,args.seed,
                                                                args.KL_coeff, args.KL_weight, args.num_samples,
                                                                args.conv_Dropout, args.drop_type, args.droprate,
                                                                args.lamb, args.lr, args.unitN, 
                                                                args.batch_size, args.nepochs, args.clip, args.drop_prior)
if args.approach == 'vcl':
    log_name = '{}_{}_{}_woDr_{}_lr_{}_KLtheta_{}_local-trick_{}_unitN_{}_batch_{}_epoch_{}'.format(args.experiment, args.approach,args.seed, 
                                                                     args.wo_Dropout, args.lr, args.KL_weight_theta, lt, args.unitN, args.batch_size, args.nepochs)


elif args.approach == 'vcl_vd':
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_KLtheta_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_lr_{}_local-trick_{}_unitN_{}_batch_{}_epoch_{}_clip_{}_drop_prior_{}'.format(
                                                                args.experiment, args.approach,args.seed,
                                                                args.KL_coeff, args.KL_weight, args.KL_weight_theta, 
                                                                args.num_samples, args.conv_Dropout, args.drop_type, args.droprate,
                                                                args.lr, lt, args.unitN, 
                                                                args.batch_size, args.nepochs, args.clip, args.drop_prior)
elif args.approach == 'ucl_vd':
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_lr_{}_local-trick_{}_lr_rho_{}_epoch_{}_clip_{}_drop_prior_{}'.format(
        args.experiment, args.approach, args.seed,  
        args.KL_coeff, args.KL_weight, args.num_samples, 
        args.conv_Dropout, args.drop_type, args.droprate, 
        args.alpha, args.beta, args.ratio, args.lr, lt, 
        args.lr_rho, args.nepochs, args.clip, args.drop_prior)


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
    args.output = './result_data/' + args.experiment + '/' + args.approach + '/' + log_name + '.txt'

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
        net_old = network.Net(inputsize, taskcla, args.ratio, unitN=args.unitN, split = split, notMNIST=notMNIST).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
    else:
        net = network.Net(inputsize, taskcla, unitN=args.unitN,  split = split, notMNIST=notMNIST).cuda()
        net_old = network.Net(inputsize, taskcla, unitN=args.unitN,  split = split, notMNIST=notMNIST).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
else:
    if args.approach == 'ucl' or args.approach == 'vcl' or args.approach == 'ucl_ablation' or args.approach == 'vcl_vd' or args.approach == 'ucl_vd':
        net = network.Net(inputsize, taskcla, args.ratio).cuda()
        net_old = network.Net(inputsize, taskcla, args.ratio).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args,
                             log_name=log_name, split=split)
    else:
        net = network.Net(inputsize, taskcla).cuda()
        net_old = network.Net(inputsize, taskcla).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args,
                             log_name=log_name, split=split)

    
num_params = utils.print_model_report(net)

# print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

# offset_label = [0]    # Increase label index for class incremental learning 
# if args.CIL != 0:
#     for i in range(1, len(taskcla)):
#         offset_label.append(offset_label[-1] + data[i]['ncla'])
# else:
#     for i in range(1, len(taskcla)):
#         offset_label.append(0)

for t, ncla in taskcla:
    
    # if t==args.tasknum:
    #     break
    
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t == 0:
            xtrain = data[t]['train']['x']
            ytrain = data[t]['train']['y']
            xvalid = data[t]['valid']['x']
            yvalid = data[t]['valid']['y']
            task_t = t * torch.ones(xtrain.size(0)).int()
            task_v = t * torch.ones(xvalid.size(0)).int()
            task = [task_t, task_v]
        else:
            xtrain = torch.cat((xtrain, data[t]['train']['x']))
            ytrain = torch.cat((ytrain, data[t]['train']['y']))
            xvalid = torch.cat((xvalid, data[t]['valid']['x']))
            yvalid = torch.cat((yvalid, data[t]['valid']['y']))
            task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
            task = [task_t, task_v]
    else:
        # Get data
        xtrain = data[t]['train']['x'].cuda()
        xvalid = data[t]['valid']['x'].cuda()
            
        ytrain = data[t]['train']['y'].cuda()
        print("ytrain size: ", ytrain.size())
        print("label size: ", ytrain[0:10])
        yvalid = data[t]['valid']['y'].cuda()
        task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save   
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')
    # torch.save(net.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))

# Print result
avg_acc, bwt = print_log_acc_bwt(acc, lss)
with open (args.output, 'a') as f:
    f.write('\n')
    f.write('avg_acc: ' + str(avg_acc) + '\n')
    f.write('bwt: ' + str(bwt) + '\n')
    f.write('Num parameters: %s'%(utils.human_format(num_params)))

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

# if hasattr(appr, 'logs'):
#     if appr.logs is not None:
#         # save task names
#         from copy import deepcopy

#         appr.logs['task_name'] = {}
#         appr.logs['test_acc'] = {}
#         appr.logs['test_loss'] = {}
#         for t, ncla in taskcla:
#             appr.logs['task_name'][t] = deepcopy(data[t]['name'])
#             appr.logs['test_acc'][t] = deepcopy(acc[t, :])
#             appr.logs['test_loss'][t] = deepcopy(lss[t, :])
#         # pickle
#         import gzip
#         import pickle

#         with gzip.open(os.path.join(appr.logpath), 'wb') as output:
#             pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
logger = logger(file_name=args.experiment + "-" + args.approach, resume=True, path='./result_data/csvresult/' + args.experiment + '/', data_format='csv')
if 'vd' in args.approach:
    logger.add(
        seed = args.seed,
        KL_weight = args.KL_weight,
        init_dr = args.droprate,
        avg_acc = avg_acc,
        file_name = log_name
        )
logger.save()
