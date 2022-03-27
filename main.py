import sys,os,argparse,time
import numpy as np
import torch

from best_hyperparams import get_best_params
from utils import *
import utils
from arguments import get_args
args = get_args()

tstart=time.time()

film=0
single_head = 0
if args.film:
    film = 1
if args.single_head:
    single_head = 1



if args.approach == 'gvclf':
    log_name = '{}_{}_{}_woDr_{}_lr_{}_film_{}_batch_{}_epoch_{}_singlehead_{}_prior_var_{}'.format(args.experiment, args.approach,args.seed, 
                                                                     args.wo_Dropout, args.lr, film, args.batch_size, args.nepochs, single_head, args.prior_var)


elif args.approach == 'gvclf_vd':
    log_name = '{}_{}_{}_KLcoeff_{}_KLweight_{}_samples_{}_conv_Dropout_{}_droptype_{}_dr_{}_lr_{}_film_{}_batch_{}_epoch_{}_singlehead_{}_prior_var_{}'.format(
                                                                args.experiment, args.approach,args.seed,
                                                                args.KL_coeff, args.KL_weight, 
                                                                args.num_samples, args.conv_Dropout, args.drop_type, args.droprate,
                                                                args.lr, film,
                                                                args.batch_size, args.nepochs, single_head, args.prior_var)
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

if args.experiment in conv_experiment:
    args.conv = True
    log_name = log_name + '_conv'

if args.output=='':
    args.output = './result_data/' + args.experiment + '/' + args.approach + '/' + log_name + '.txt'
print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################
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
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader
elif args.experiment=='easy-chasy':
    from dataloaders import easy_chasy as dataloader
elif args.experiment=='hard-chasy':
    from dataloaders import hard_chasy as dataloader
elif args.experiment=='smnist':
    from dataloaders import smnist as dataloader
elif args.experiment=='split_mnist':
    from dataloaders import split_mnist as dataloader
elif args.experiment == 'split_cifar100':
    from dataloaders import split_cifar100 as dataloader
elif args.experiment == 'split_cifar10_100':
    from dataloaders import split_cifar10_100 as dataloader
elif args.experiment == 'omniglot':
    from dataloaders import split_omniglot as dataloader
elif args.experiment == 'pmnist':
    from dataloaders import pmnist as dataloader

# Args -- Approach
if args.approach=='random':
    from approaches import random as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach=='sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach=='lwf':
    from approaches import lwf as approach
elif args.approach=='lfl':
    from approaches import lfl as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='ewc-film':
    from approaches import ewc_film as approach
elif args.approach=='ewc2':
    from approaches import ewc2 as approach
elif args.approach=='imm-mean':
    from approaches import imm_mean as approach
elif args.approach=='imm-mode':
    from approaches import imm_mode as approach
elif args.approach=='progressive':
    from approaches import progressive as approach
elif args.approach=='pathnet':
    from approaches import pathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat as approach
elif 'gvclf_vd' == args.approach:
    from approaches import gvclf_vd as approach
elif 'gvclf' == args.approach:
    from approaches import gvclf as approach
elif args.approach=='joint':
    from approaches import joint as approach
elif 'vcl' in args.approach:
    from approaches import gvclf as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist':
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    elif 'gvclf' == args.approach:
        from networks.gvcl_models import MLPFilm as network
    elif 'gvclf_vd' in args.approach:
        from networks.gvcl_models import MLPFilmVD as network
    else:
        from networks import mlp as network
elif args.experiment == 'mixture':
    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='ewc-film':
        from networks import alexnet_ewc_film as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
    elif 'vclf' in args.approach:
        from networks.gvcl_models import AlexNetFiLM as network
    elif 'vcl' in args.approach:
        from networks.gvcl_models import AlexNetNoFiLM as network
    else:
        from networks import alexnet as network
    

        
elif args.experiment == 'cifar' or args.experiment == 'split_cifar100' or args.experiment == 'split_cifar10_100':
    if args.approach=='lfl':
        from networks import zenkenet_lfl as network
    elif args.approach=='hat':
        from networks import zenkenet_hat as network
    elif args.approach=='progressive':
        from networks import zenkenet_progressive as network
    elif args.approach=='pathnet':
        from networks import zenkenet_pathnet as network
    elif args.approach=='hat-test':
        from networks import zenkenet_hat_test as network
    elif args.approach=='ewc-film':
        from networks import zenkenet_ewc_film as network
    elif 'gvclf' == args.approach:
        from networks.gvcl_models import CNNFilm as network
    elif 'gvclf_vd' == args.approach:
        from networks.gvcl_models import CNNFilmVD as network
    elif 'gvcl' == args.approach:
        from networks.gvcl_models import ZenkeNetNoFiLM as network
    else:   
        from networks import zenkenet as network

elif 'chasy' in args.experiment:
    if args.approach=='lfl':
        from networks import babynet_lfl as network
    elif args.approach=='hat':
        from networks import babynet_hat as network
    elif args.approach=='progressive':
        from networks import babynet_progressive as network
    elif args.approach=='pathnet':
        from networks import babynet_pathnet as network
    elif args.approach=='ewc-film':
        from networks import babynet_ewc_film as network
    elif 'vclf' in args.approach:
        from networks.gvcl_models import BabyNetFiLM as network
    elif 'vcl' in args.approach:
        from networks.gvcl_models import BabyNetNoFiLM as network
    else:
        from networks import babynet as network

elif 'smnist' == args.experiment or 'split_mnist' == args.experiment:
    if args.approach=='lfl':
        from networks import smnistnet_lfl as network
    elif args.approach=='hat':
        from networks import smnistnet_hat as network
    elif args.approach=='progressive':
        from networks import smnistnet_progressive as network
    elif args.approach=='pathnet':
        from networks import smnistnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import smnistnet_hat_test as network
    elif args.approach=='ewc-film':
        from networks import smnistnet_ewc_film as network
    elif args.approach=='ewc2':
        from networks import smnistnet_binary as network
    elif 'gvclf' == args.approach:
        from networks.gvcl_models import MLPFilm as network
    elif 'gvclf_vd' in args.approach:
        from networks.gvcl_models import MLPFilmVD as network
    else:
        from networks import smnistnet as network

elif args.experiment == 'omniglot':
    if 'gvclf' == args.approach:
        from networks.gvcl_models import CNNOmniglotFilm as network
    elif 'gvclf_vd' in args.approach:
        from networks.gvcl_models import CNNOmniglotFilmVD as network
########################################################################################################################

# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(seed=args.seed)
if args.ntasks != -1:
    taskcla = taskcla[:args.ntasks]
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(inputsize,taskcla).cuda()
utils.print_model_report(net)

#Set hyperparameters
best_param, best_lr, best_epochs = get_best_params(args.approach, args.experiment)
if args.nepochs == -1:
    args.nepochs = best_epochs
    print("using default # epochs of {}".format(best_epochs))
if args.lr == -1:
    args.lr = best_lr
    print("using default lr of {}".format(best_lr))
if len(args.parameter) == 0:
    args.parameter = best_param
    print("using default hyperparams of {}".format(best_param))

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args, sbatch = args.batch_size)

print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop taskki,l
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()
        if args.approach == 'hat':
            test_loss,test_acc=appr.eval(u,xtest,ytest,save_preds = True, dset = args.experiment)
        else:
            test_loss,test_acc=appr.eval(u,xtest,ytest,)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f')

# Print result
avg_acc, bwt = print_log_acc_bwt(acc, lss)
with open (args.output, 'a') as f:
    f.write('\n')
    f.write('avg_acc: ' + str(avg_acc) + '\n')
    f.write('bwt: ' + str(bwt) + '\n')

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
logger = logger(file_name=args.experiment + "-" + args.approach, resume=True, path='./result_data/csvdata/' + args.experiment + '/', data_format='csv')
if 'vd' in args.approach:
    logger.add(film = film,
        KL_weight = args.KL_weight,
        init_dr = args.droprate,
        beta = float(args.parameter.split(',')[0]),
        lamb = float(args.parameter.split(',')[1]),
        avg_acc = avg_acc,
        file_name = log_name
        )
else:
    logger.add(
        film = film,
        beta = float(args.parameter.split(',')[0]),
        lamb = float(args.parameter.split(',')[1]),
        avg_acc = avg_acc,
        file_name = log_name,
        )
logger.save()


