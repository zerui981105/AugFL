import argparse
import torch
import numpy as np
import random

DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1','cifar10','cifar100','Fmnist']
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('--optimizer',default='HFfmaml',help='name of optimizer;',type=str,choices=OPTIMIZERS)
    parser.add_argument('--dataset',default='cifar100_100',help='name of dataset;',type=str,choices=DATASETS)
    parser.add_argument('--model',default='cnn',help='name of model;',type=str)
    parser.add_argument('--num_rounds',default=600,help='number of rounds to simulate;',type=int)
    parser.add_argument('--eval_every',default=6,help='evaluate every rounds;',type=int)
    parser.add_argument('--clients_per_round',default=80,help='number of clients trained per round;',type=int)
    parser.add_argument('--batch_size',default=100,help='batch size when clients train on data;',type=int)
    parser.add_argument('--num_epochs',default=1,help='number of epochs when clients train on data;',type=int) #20
    parser.add_argument('--alpha',default=0.02,help='learning rate for inner solver;',type=float)
    parser.add_argument('--beta',default=0.003,help='meta rate for inner solver;',type=float)
    # parser.add_argument('--mu',help='constant for prox;',type=float,default=0.01)
    parser.add_argument('--seed',default=0,help='seed for randomness;',type=int)
    parser.add_argument('--labmda',default=0,help='labmda for regularizer',type=float)
    parser.add_argument('--rho',default=1.5,help='rho for regularizer',type=float)
    parser.add_argument('--mu_i',default=0,help='mu_i for optimizer',type=int)
    parser.add_argument('--adapt_num', default=1, help='adapt number', type=int)
    parser.add_argument('--isTrain', default=True, help='load trained wights', type=bool)
    parser.add_argument('--pretrain', default=False, help='Pretrain to get theta_c', type=bool)
    parser.add_argument('--sourceN', default=10, help='source node class num used', type=int)
    parser.add_argument('--R', default=0, help='the R th test', type=int)
    parser.add_argument('--logdir', default='./log', help='the R th test', type=str)
    parser.add_argument('--transfer', default=False, help='Pretrain to get theta_c', type=bool)
    parser.add_argument('--device', type = str, default='cuda', help='device')
    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    torch.manual_seed(123 + parsed['seed'])
    torch.cuda.manual_seed_all(123 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    random.seed(1 + parsed['seed'])

    return parsed