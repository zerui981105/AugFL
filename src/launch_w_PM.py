import numpy as np
import argparse
import random
import os
import json
from baseModel_cifar100 import BaseModel as BaseModel_c100
import torch
from ServerMain import (Server)
from torch.nn import init
from torch import nn
import socket
from CRD.crd_interface import get_teacher_name, run_CRD

DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1','cifar10','cifar100','Fmnist']



def read_options_kd():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default='./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth', help='teacher model snapshot')
    # parser.add_argument('--path_t', type=str, default='./save/models/resnet110_cifar10_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth', help='teacher model snapshot')
    # parser.add_argument('--path_t', type=str,
    #                     default='./save/models/resnet32x4_fmnist_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth',
    #                     help='teacher model snapshot')
    # parser.add_argument('--path_t', type=str,
    #                     default='./save/models/resnet32x4_emnist_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth',
    #                     help='teacher model snapshot')
    # distillation
    parser.add_argument('--distill', type=str, default='crd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.8, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    # parser.add_argument('--nce_k', default=5000, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('--optimizer',default='HFfmaml',help='name of optimizer;',type=str,choices=OPTIMIZERS)
    parser.add_argument('--dataset',default='cifar100',help='name of dataset;',type=str,choices=DATASETS)
    parser.add_argument('--model',default='cnn',help='name of model;',type=str)
    parser.add_argument('--num_rounds',default=301,help='number of rounds to simulate;',type=int)
    parser.add_argument('--eval_every',default=10,help='evaluate every rounds;',type=int)
    parser.add_argument('--clients_per_round',default=40,help='number of clients trained per round;',type=int)
    parser.add_argument('--batch_size',default=100,help='batch size when clients train on data;',type=int)
    parser.add_argument('--num_epochs',default=1,help='number of epochs when clients train on data;',type=int) #20
    parser.add_argument('--alpha',default=0.05,help='learning rate for inner solver;',type=float)
    parser.add_argument('--beta',default=0.01,help='meta rate for inner solver;',type=float)
    # parser.add_argument('--mu',help='constant for prox;',type=float,default=0.01)
    parser.add_argument('--seed',default=0,help='seed for randomness;',type=int)
    parser.add_argument('--labmda',default=5.0,help='labmda for regularizer',type=float)
    parser.add_argument('--rho',default=0.7,help='rho for regularizer',type=float)
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

def reshape_label(label,n=26):
    #print(label)
    new_label=[0]*n
    new_label[int(label)]=1
    return new_label

def reshapeFmnist(x):
    x=np.array(x)
    x=x.reshape(28,28,1)
    x = np.transpose(x, [2,0,1])
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
    if options['dataset']=='cifar10' or options['dataset']=='cifar100' or options['dataset']=='cifar10_20':
        train_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'train')
        test_path = os.path.join('/root/autodl-tmp/fmaml/fmaml_mac/data', options['dataset'], 'data', 'test')
        if options['pretrain'] or situation=='forget_test':
            print('@@@@@@@@@@@@@@@@@using pretrained dataset')
            train_path = os.path.join('data', options['dataset'], 'data', 'pretrain')
            test_path = os.path.join('data', options['dataset'], 'data', 'pretest')
        dataset = read_data(train_path, test_path)
        # target_dataset = read_data(target_train_path, target_test_path)
        num_class = 10
        if options['dataset'] == 'cifar100' or options['dataset']=='cifar100_100':
            num_class = 100
        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i]=reshape_features(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i],num_class)
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



    else:
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        dataset = read_data(train_path, test_path)
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
    opt = read_options_kd()
    print('train with {}'.format(args['dataset']))

    test_users, dataset = prepare_dataset(args)
    s = Server(args, BaseModel_c100, dataset, test_users, opt)

    s.train_CRD()



if __name__ == '__main__':
    main()