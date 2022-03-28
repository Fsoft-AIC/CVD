import argparse

def get_args():
    parser=argparse.ArgumentParser(description='Continual')
    parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
    parser.add_argument('--experiment',default='',type=str,required=True,choices=['mnist2','pmnist','cifar','mixture', 'easy-chasy', 'hard-chasy', 'smnist', 'split_mnist', 'split_cifar100', 'split_cifar10_100', 'omniglot', 'pmnist'],help='(default=%(default)s)')
    parser.add_argument('--approach',default='',type=str,required=True,choices=['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                                'imm-mode','sgd-restart', 'ewc2', 'ewc-film',
                                                                                'joint','hat','hat-test', 'gvcl', 'vcl', 'vclf', 'gvclf', 'gvclf_vd'],help='(default=%(default)s)')
    parser.add_argument('--film', action='store_true', default=False, help='Add film layers')
    parser.add_argument('--batch-size', default=64, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--nepochs',default=-1,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--lr',default=-1,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--ntasks',type=int,default=-1,help='(default=%(default)s)')
    parser.add_argument('--use-best-hyperparams',type=bool,default=True,help='(default=%(default)s)')
    parser.add_argument('--drop_type', type=str, default='Gauss', choices= ['Gauss', 'AddNoise'], help='Choose type of dropout') 
    parser.add_argument('--droprate', type=float, default=0.5, help="Choose dropout rate") 
    parser.add_argument('--droprate_linear', type=float, default=0.5, help="Choose dropout rate for linear net") 
    parser.add_argument('--single_head', action='store_true', default=False, help='Use single head')
    parser.add_argument('--wo_Dropout', action='store_true', default=False, help="without Dropout exp") 
    parser.add_argument('--conv_Dropout', action='store_true', default=False, help="VD for CNN") 
    parser.add_argument('--num_samples', type=int, default=1, help='The number of taking sample') 
    parser.add_argument('--KL_coeff', type=str, choices=['1', '1_M', '1_N', 'M_N'], default='1_N', required=False) 
    parser.add_argument('--KL_weight', type=float, default=1, help='Choose the KL weight of noise') 
    parser.add_argument('--test_samples', type=int, default=100, help='Number of sample at test time') 
    parser.add_argument('--prior_var',default=-1,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--equalize_epochs',type=bool,default=False,help='(default=%(default)s)')

    args=parser.parse_args()

    return args