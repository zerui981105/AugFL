import argparse
import torch
import numpy as np
import random
import numpy as np
import argparse
import importlib
import random
import os
import json
from tqdm import  tqdm
from baseModel import BaseModel
from baseModel_cifar100 import BaseModel as BaseModel_c100
from baseModel_stl10 import BaseModel as BaseModel_stl10
from baseModel_fmnist import BaseModel as BaseModel_fmn
#import pandas as pd
import torch
from PIL import Image
# from options_arg import read_options
# from ServerMain import Server

from ServerMain import (Server)
import wandb
import uuid
from clientbase import Client
from torch.nn import init
from torch import nn

DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1','cifar10','cifar100','Fmnist']
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('--optimizer',default='HFfmaml',help='name of optimizer;',type=str,choices=OPTIMIZERS)
    parser.add_argument('--dataset',default='emnist',help='name of dataset;',type=str,choices=DATASETS)
    parser.add_argument('--model',default='cnn',help='name of model;',type=str)
    parser.add_argument('--num_rounds',default=300,help='number of rounds to simulate;',type=int)
    parser.add_argument('--eval_every',default=10,help='evaluate every rounds;',type=int)
    parser.add_argument('--clients_per_round',default=80,help='number of clients trained per round;',type=int)
    parser.add_argument('--batch_size',default=100,help='batch size when clients train on data;',type=int)
    parser.add_argument('--num_epochs',default=1,help='number of epochs when clients train on data;',type=int) #20
    parser.add_argument('--alpha',default=0.02,help='learning rate for inner solver;',type=float)
    parser.add_argument('--beta',default=0.015,help='meta rate for inner solver;',type=float)
    # parser.add_argument('--mu',help='constant for prox;',type=float,default=0.01)
    parser.add_argument('--seed',default=0,help='seed for randomness;',type=int)
    parser.add_argument('--labmda',default=0,help='labmda for regularizer',type=float)
    parser.add_argument('--rho',default=1.0,help='rho for regularizer',type=float)
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

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def reshape_label(label,n=47):
    #print(label)
    new_label=[0]*n
    new_label[int(label)]=1
    return new_label

def reshapeFmnist(x):
    x=np.array(x)
    x=x.reshape(28,28,1)
    x = np.transpose(x, [2, 0, 1])
    return x

def reshapeStl10(x):
    x=np.array(x)
    x=x.reshape(3,96,96)
    return x

def reshape_features(x):
    x=np.array(x)
    x = x.reshape(3, 32, 32)
    # x = np.transpose(x.reshape(3, 32, 32), [1, 2, 0])
    # x = np.transpose(x.reshape(32, 32, 3), [2, 0, 1])
    #print(x.shape)
    return x

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.trunc_normal_(m.weight.data, mean=0.5,std=1.0,a=-2.0,b=2.0)
        init.constant_(m.bias.data,1.0)
    elif isinstance(m, nn.Linear):
        init.trunc_normal_(m.weight.data, std=1.0,a=-2.0,b=2.0)
        init.constant_(m.bias.data, 1.0)


def load_data(options, situation='normal_train'):
    # path = '/root/autodl-tmp/fmaml_mac/data/cifar10'
    path = '/root/autodl-tmp/fmaml/fmaml_mac/data/cifar10'
    train_path = os.path.join(path, 'data', 'train')
    test_path = os.path.join(path, 'data', 'test')
    dataset = read_data(train_path, test_path)
    num_class = 10

    for user in dataset[0]:
        for i in range(len(dataset[2][user]['y'])):
            dataset[2][user]['x'][i] = reshape_features(dataset[2][user]['x'][i])
            dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i], num_class)

    # print('reshape labels in test dataset')
    for user in dataset[0]:
        for i in range(len(dataset[3][user]['y'])):
            dataset[3][user]['x'][i] = reshape_features(dataset[3][user]['x'][i])
            dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i], num_class)

    random.seed(1)
    random.shuffle(dataset[0])
    test_user = dataset[0][options['clients_per_round']:]
    del dataset[0][options['clients_per_round']:]
    return test_user, dataset

def prepare_dataset(options,situation='normal_train'):
    # read data
    if options['dataset']=='cifar10' or options['dataset']=='cifar100' or options['dataset']=='cifar100_100':
        # data_path = os.path.join('data', options['dataset'], 'data')
        # dataset = read_data_xin(data_path)  # return clients, groups, train_data, test_data
        train_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'train')
        test_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'test')
        if options['pretrain'] or situation=='forget_test':
            print('@@@@@@@@@@@@@@@@@using pretrained dataset')
            train_path = os.path.join('data', options['dataset'], 'data', 'pretrain')
            test_path = os.path.join('data', options['dataset'], 'data', 'pretest')
        dataset = read_data(train_path, test_path)
        num_class = 10
        if options['dataset'] == 'cifar100' or options['dataset']=='cifar100_100':
            num_class = 100
        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i]=reshape_features(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i],num_class)

        # print('reshape labels in test dataset')
        for user in dataset[0]:
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['x'][i] = reshape_features(dataset[3][user]['x'][i])
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i],num_class)

    elif options['dataset']=='Fmnist' or options['dataset']=='emnist':
        train_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'train')
        test_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'test')
        if options['pretrain'] or situation=='forget_test':
            print('@@@@@@@@@@@@@@@@@using pretrained dataset')
            train_path = os.path.join('data', options['dataset'], 'data', 'pretrain')
            test_path = os.path.join('data', options['dataset'], 'data', 'pretest')
        dataset = read_data(train_path, test_path) # return clients, groups, train_data, test_data
        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i] = reshapeFmnist(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['x'][i] = reshapeFmnist(dataset[3][user]['x'][i])
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])


    elif options['dataset']=='stl10':
        train_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'train')
        test_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'test')
        if options['pretrain'] or situation=='forget_test':
            print('@@@@@@@@@@@@@@@@@using pretrained dataset')
            train_path = os.path.join('data', options['dataset'], 'data', 'pretrain')
            test_path = os.path.join('data', options['dataset'], 'data', 'pretest')
        dataset = read_data(train_path, test_path) # return clients, groups, train_data, test_data
        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i] = reshapeStl10(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['x'][i] = reshapeStl10(dataset[3][user]['x'][i])
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])

    else:
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        dataset = read_data(train_path, test_path) # return clients, groups, train_data, test_data
        #print(dataset[3]['f_00000']['y'])
        #print('@main_HFfaml.py line 152####',dataset)

        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])

    ######************************************** devide  source node and target node **********************************************#########################
    random.seed(1)
    random.shuffle(dataset[0])
    test_user=dataset[0][options['clients_per_round']:]
    del dataset[0][options['clients_per_round']:]
    return test_user, dataset
def main():


    args = read_options()
    torch.manual_seed(123 + args['seed'])
    torch.cuda.manual_seed_all(123 + args['seed'])
    np.random.seed(12 + args['seed'])
    random.seed(1 + args['seed'])
    print('train with {}'.format(args['dataset']))

    # torch.manual_seed(123 + args['seed'])
    # torch.cuda.manual_seed_all(123 + args['seed'])
    # np.random.seed(12 + args['seed'])
    # random.seed(1 + args['seed'])

    test_users, dataset = prepare_dataset(args)
    s = Server(args, BaseModel, dataset, test_users)
    # for i in range(500):
    #     a,b, c,d = s.clients[0].train()
    #     print(a)
    #     print(b)
    #     print(c)

    s.train_maml()
    # users, groups, train_data, test_data = dataset
    # if len(groups) == 0:
    #     groups = [None for _ in users]
    # all_clients = []
    # total_sample_num = 0
    # w_is = []
    # for u, g in zip(users, groups):
    #     num_i = len(train_data[u]['y']) + len(test_data[u]['y'])
    #     w_is.append(num_i)
    #     total_sample_num += num_i
    # w_is = [x / total_sample_num for x in w_is]
    # #
    # #     # create clients
    # # args['w_i'] = w_is[0]
    # # model = BaseModel()
    # # c = Client(users[0], train_data[users[0]], test_data[users[0]], args, model)
    # # for i in range(500):
    # #     a,b,d = c.train_avg()
    # #     print(a)
    # #     print(b)
    # for u, g, w_i in zip(users, groups, w_is):
    #     args['w_i'] = w_i
    #     model = BaseModel()
    #     # model = resnet8x4(num_classes=10)
    #     all_clients.append(Client(u, train_data[u], test_data[u], args, model))
    # global_model = BaseModel()
    # for i in range(500):
    #     print(i)
    #     loss = 0
    #     acc = 0
    #     for c in all_clients:
    #         ac = 0
    #         lo = 0
    #         for t in range(10):
    #             a,b,d = c.train_avg()
    #             ac += a
    #             lo += b
    #         ac /= 10
    #         lo /= 10
    #         acc += ac
    #         loss += lo
    #     print(acc / 40)
    #     print(loss / 40)
    #     for j in range(len(all_clients)):
    #         for main_param, agent_param in zip(global_model.parameters(),
    #                                            all_clients[j].model.parameters()):
    #             if (j == 0):
    #                 main_param.data.copy_(agent_param)
    #             else:
    #                 main_param.data.copy_(main_param * (j / (j + 1)) + agent_param * (1 / (j + 1)))
    #     for z in range(len(all_clients)):
    #         for main_agent_param, agent_param in zip(global_model.parameters(), all_clients[z].model.parameters()):
    #             agent_param.data.copy_(main_agent_param)




    # for u, g, w_i in zip(users, groups, w_is):
    #     params['w_i'] = w_i
    #     model = self.learner(params)
    #     all_clients.append(Client(u, g, train_data[u], test_data[u], model))

if __name__ == '__main__':
    main()