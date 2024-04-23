import torch
import os
import numpy as np
# import h5py
import copy
import time
import random
from tqdm import trange, tqdm
import wandb
import uuid
from torch import nn
from torch.nn import init
from CRD.crd_interface import run_CRD,CRD

# from utils.data_utils import read_client_data
# from utils.dlg import DLG

from clientbase import Client
from resnet import resnet8x4

project: str = "ADMM"
group: str = "ADMM_meta"
name: str = "crd"
def wandb_init() -> None:
    wandb.init(
        project=project,
        group=group,
        name=name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
def target_test2(test_user,learner,dataset,options,weight):
    accs=dict()
    num_test=dict()
    for i,user in enumerate(test_user):
        accs[i], num_test[i]=final_test(learner=learner, train_data=dataset[2][user], test_data=dataset[3][user],
                params=options, user_name=user, weight= weight)
    accs=list(accs.values())
    num_test=list(num_test.values())
    acc_test = [a * n/np.sum(num_test) for a, n in zip(accs, num_test)]
    return np.sum(acc_test)

def final_test(learner, train_data, test_data, params, user_name, weight):
    # print('HFmaml test')
    params['w_i']=1
    # client_model = learner()  # changed remove star
    client_model = resnet8x4(num_classes=100)  # changed remove star
    test_client = Client(user_name, train_data, test_data, params, client_model)
    test_client.set_parameters(weight)
    _ , num_test = test_client.fast_adapt()
    return _, num_test



# def weight_init(m):
#     if isinstance(m, nn.Conv2d):
#         init.trunc_normal_(m.weight.data, std=0.1, a=-0.2,b=0.2)
#         # init.constant_(m.weight.data, 0)
#         init.constant_(m.bias.data,0.1)
#     elif isinstance(m, nn.Linear):
#         init.trunc_normal_(m.weight.data, std=0.1,a=-0.2, b=0.2)
#         # init.constant_(m.weight.data, 0)
#         init.constant_(m.bias.data, 0.1)

class Server(object):
    def __init__(self, params, learner, datasets, test_user,opt):
        # Set up the main attributes
        self.opt = opt
        self.params = params
        # self.lamda = params['labmda']
        self.transfer = params['transfer']
        self.test_user = test_user
        self.datasets_data = datasets
        _, _, self.train_data, self.test_data = datasets
        for key, val in params.items(): setattr(self, key, val);
        _, _, train_data, test_data = datasets
        params['w_i'] = 1
        self.learner = learner
        self.clients = self.set_clients(datasets, params)
        # self.global_model = learner()
        self.global_model = resnet8x4(num_classes=100)

    def set_clients(self, dataset, params):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients=[]
        total_sample_num=0
        w_is=[]
        for u, g in zip(users, groups):
            num_i=len(train_data[u]['y'])+len(test_data[u]['y'])
            print(num_i)
            w_is.append(num_i)
            total_sample_num+=num_i
        w_is=[x/total_sample_num for x in w_is]

        for u, g, w_i in zip(users, groups, w_is):
            params['w_i'] = w_i
            # model = self.learner()
            model = resnet8x4(num_classes=100)
            all_clients.append(Client(u, train_data[u], test_data[u], params,model))
        # print(all_clients[0].model.state_dict()['conv_1.bias'])
        return all_clients



    def select_clients(self, round, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)
        return np.random.choice(self.clients, num_clients, replace=False)

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)



    # def receive_models(self):
    #     assert (len(self.selected_clients) > 0)
    #
    #     active_clients = random.sample(
    #         self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
    #
    #     self.uploaded_ids = []
    #     self.uploaded_weights = []
    #     self.uploaded_models = []
    #     tot_samples = 0
    #     for client in active_clients:
    #         try:
    #             client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
    #                     client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
    #         except ZeroDivisionError:
    #             client_time_cost = 0
    #         if client_time_cost <= self.time_threthold:
    #             tot_samples += client.train_samples
    #             self.uploaded_ids.append(client.id)
    #             self.uploaded_weights.append(client.train_samples)
    #             self.uploaded_models.append(client.model)
    #     for i, w in enumerate(self.uploaded_weights):
    #         self.uploaded_weights[i] = w / tot_samples
    # def aggregate_avg(self):
    #     self.uploaded_models = []
    #     for client in self.clients:
    #         self.uploaded_models.append(client.model)
    #     for param in self.global_model.parameters():
    #         param.data.zero_()
    #     n = 0
    #     for client in self.clients:
    #         n += client.num_samples
    #     n_w = []
    #     for c in self.clients:
    #         n_w.append(c.num_samples / n)
    #     for w, c_model in zip(n_w, self.uploaded_models):
    #         for server_param, client_param in zip(self.global_model.parameters(), c_model.parameters()):
    #             server_param.data += w * client_param.data.clone()

    def receive_models(self):
        assert (len(self.clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.clients:
            # try:
            #     client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            # except ZeroDivisionError:
            #     client_time_cost = 0
            # if client_time_cost <= self.time_threthold:
            tot_samples += client.num_train
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.num_train)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_avg(self):
        assert (len(self.clients) > 0)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_avg(w, client_model)

    def add_parameters_avg(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters_CRD(self,grad):
        self.uploaded_ys = []
        self.uploaded_models = []
        for client in self.clients:
            self.uploaded_ys.append(client.yy_k)
            self.uploaded_models.append(client.model)

        # self.global_model = copy.deepcopy(self.clients[0].model)
        for param in self.global_model.parameters():
            param.data.zero_()
        n = len(self.clients)
        sum_rho = self.rho * n
        for y, client_model in zip(self.uploaded_ys, self.uploaded_models):
            self.add_parameters(y, client_model)
        for p , CRD_grad in zip(self.global_model.parameters(), grad):
            p.data = (p.data - self.labmda*CRD_grad)/(sum_rho)

    def aggregate_parameters(self):
        self.uploaded_ys = []
        self.uploaded_models = []
        for client in self.clients:
            self.uploaded_ys.append(client.yy_k)
            self.uploaded_models.append(client.model)

        # self.global_model = copy.deepcopy(self.clients[0].model)
        for param in self.global_model.parameters():
            param.data.zero_()
        n = len(self.clients)
        sum_rho = self.rho * n
        for y, client_model in zip(self.uploaded_ys, self.uploaded_models):
            self.add_parameters(y, client_model)
        for p in self.global_model.parameters():
            p.data = p.data / sum_rho

    def add_parameters(self, y, client_model):
        for server_param, client_param, sy in zip(self.global_model.parameters(), client_model.parameters(), y):
            server_param.data += sy + self.rho * client_param.data.clone()


    def train_avg(self,local_epoch):
        print('Training with {} workers ---'.format(self.clients_per_round))
        wandb.init()
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            if i % self.eval_every==0:
                # train_acc_sum = []
                train_acc_sum = []
                loss_sum = []
                num_sum = []
                for ci, c in enumerate(self.clients):
                    train_acc, train_loss, num_samples = c.train_avg(local_epoch)

                    loss_sum.append(train_loss.detach().numpy())
                    train_acc_sum.append(train_acc.detach().numpy())
                    num_sum.append(num_samples)
                tot_sams = np.sum(num_sum)
                train_acc_mean = [ n / tot_sams * test_acc for n, test_acc in zip(num_sum, train_acc_sum)]
                loss_mean = [n / tot_sams * loss for n, loss in zip(num_sum, loss_sum)]

                target_acc = target_test2(self.test_user, self.learner, self.datasets_data, self.params, self.global_model)
                wandb.log(
                    {"target_eval": target_acc},
                    step=i,
                )
                tqdm.write(
                    'At round {} training loss: {};  acc_test:{}, target acc:{}'.format(i, np.sum(loss_mean),np.sum(train_acc_mean), target_acc))
                self.receive_models()
                self.aggregate_parameters_avg()
                # self.aggregate_avg()
                self.send_models()
            else:
                for ci,c in enumerate(self.clients):
                    test_acc, train_loss_1, num_samples = c.train_avg(local_epoch)
                # self.aggregate_avg()
                # self.send_models()
                self.receive_models()
                self.aggregate_parameters_avg()
                # self.aggregate_avg()
                self.send_models()

    def train_maml(self, local_epoch):
        print('Training with {} workers ---'.format(self.clients_per_round))
        wandb.init()
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            if i % self.eval_every==0:
                # train_acc_sum = []
                train_acc_sum = []
                loss_sum = []
                num_sum = []
                for ci, c in enumerate(self.clients):
                    train_acc, train_loss, num_samples = c.train_maml(local_epoch)

                    loss_sum.append(train_loss.detach().numpy())
                    train_acc_sum.append(train_acc.detach().numpy())
                    num_sum.append(num_samples)
                tot_sams = np.sum(num_sum)
                train_acc_mean = [ n / tot_sams * test_acc for n, test_acc in zip(num_sum, train_acc_sum)]
                loss_mean = [n / tot_sams * loss for n, loss in zip(num_sum, loss_sum)]

                target_acc = target_test2(self.test_user, self.learner, self.datasets_data, self.params, self.global_model)
                wandb.log(
                    {"target_eval": target_acc},
                    step=i,
                )
                tqdm.write(
                    'At round {} training loss: {};  acc_test:{}, target acc:{}'.format(i, np.sum(loss_mean),np.sum(train_acc_mean), target_acc))
                self.receive_models()
                self.aggregate_parameters_avg()
                # self.aggregate_avg()
                self.send_models()
            else:
                for ci,c in enumerate(self.clients):
                    test_acc, train_loss_1, num_samples = c.train_maml(local_epoch)
                # self.aggregate_avg()
                # self.send_models()
                self.receive_models()
                self.aggregate_parameters_avg()
                # self.aggregate_avg()
                self.send_models()

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        wandb.init(project='ADMM',
                   group='ADMM_meta',
                   name='target_acc_num_cifar10',
                   config=self.params)
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            if i % self.eval_every == 0:
                train_acc_sum = []
                test_acc_sum = []
                loss_sum = []
                num_sum = []
                for ci, c in enumerate(self.clients):
                    train_acc, test_acc, train_loss_1, num_samples = c.train()
                    # c.update_yyk()

                    train_acc_sum.append(train_acc)
                    test_acc_sum.append((test_acc))
                    loss_sum.append(train_loss_1)
                    num_sum.append(num_samples)
                tot_sams = np.sum(num_sum)
                train_acc_mean = [n / tot_sams * train_acc for n, train_acc in zip(num_sum, train_acc_sum)]
                test_acc_mean = [n / tot_sams * test_acc for n, test_acc in zip(num_sum, test_acc_sum)]
                loss_mean = [n / tot_sams * loss for n, loss in zip(num_sum, loss_sum)]

                target_acc = target_test2(self.test_user, self.learner, self.datasets_data, self.params,
                                          self.global_model)
                wandb.log(
                    {"target_eval": target_acc},
                    step=i,
                )
                # target_acc = None
                tqdm.write(
                    'At round {} training loss: {}; acc_train:{}; acc_test:{}, target acc:{}'.format(i,
                                                                                                     np.sum(loss_mean),
                                                                                                     np.sum(
                                                                                                         train_acc_mean),
                                                                                                     np.sum(
                                                                                                         test_acc_mean),
                                                                                                     target_acc))
                self.aggregate_parameters()
                # print(self.global_model.state_dict()['conv_1.weight'])
                # print('%%%%%%%%%%%%%%%%%%%%%%%')
                self.send_models()
            else:
                for ci, c in enumerate(self.clients):
                    train_acc, test_acc, train_loss_1, num_samples = c.train()
                    # c.update_yyk()
                self.aggregate_parameters()
                self.send_models()

    def crd_single(self):
        wandb.init()
        run_CRD = CRD(self.opt, self.global_model)
        for i in range(300):
            print(i)
            grad = run_CRD.train(self.global_model)
            device = torch.device("cpu")
            self.global_model = self.global_model.to(device)
            for weight, phy in zip(self.global_model.parameters(), grad):
                weight.data = (weight.data - self.alpha * phy)
            if i % 10 == 0:
                target_acc = target_test2(self.test_user, self.learner, self.datasets_data, self.params, self.global_model)
                wandb.log(
                    {"crd_single": target_acc},
                    step=i,
                )
                print(target_acc)
    def train_CRD(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        # wandb.init()
        wandb.init(project='ADMM',
                   group='ADMM_meta',
                   name='target_acc_admm_crd_cifar10',
                   config=self.params)
        run_CRD = CRD(self.opt, self.global_model)
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            if i % self.eval_every==0:
                train_acc_sum = []
                test_acc_sum = []
                loss_sum = []
                num_sum = []
                target_acc = target_test2(self.test_user, self.learner, self.datasets_data, self.params,self.global_model)
                grad = run_CRD.train(self.global_model)
                device = torch.device("cpu")
                self.global_model = self.global_model.to(device)
                for ci, c in enumerate(self.clients):
                    train_acc, test_acc, train_loss_1, num_samples = c.train()
                    # c.update_yyk()

                    train_acc_sum.append(train_acc)
                    test_acc_sum.append((test_acc))
                    loss_sum.append(train_loss_1)
                    num_sum.append(num_samples)
                tot_sams = np.sum(num_sum)
                train_acc_mean = [ n / tot_sams * train_acc for n, train_acc in zip(num_sum, train_acc_sum)]
                test_acc_mean = [ n / tot_sams * test_acc for n, test_acc in zip(num_sum, test_acc_sum)]
                loss_mean = [n / tot_sams * loss for n, loss in zip(num_sum, loss_sum)]

                # target_acc = target_test2(self.test_user, self.learner, self.datasets_data, self.params, self.global_model)
                wandb.log(
                    {"target_eval": target_acc},
                    step=i,
                )
                # target_acc = None
                tqdm.write(
                    'At round {} training loss: {}; acc_train:{}; acc_test:{}, target acc:{}'.format(i, np.sum(loss_mean),np.sum(train_acc_mean),np.sum(test_acc_mean), target_acc))
                self.aggregate_parameters_CRD(grad)
                # run_CRD(self.opt, self.global_model)
                # run_CRD.train(self.global_model)
                # print(self.global_model.state_dict()['conv_1.weight'])
                # print('%%%%%%%%%%%%%%%%%%%%%%%')
                # device = torch.device("cpu")
                # self.global_model = self.global_model.to(device)
                self.send_models()
            else:

                grad = run_CRD.train(self.global_model)
                device = torch.device("cpu")
                self.global_model = self.global_model.to(device)
                for ci,c in enumerate(self.clients):
                    train_acc, test_acc, train_loss_1, num_samples = c.train()
                    # c.update_yyk()
                self.aggregate_parameters_CRD(grad)
                # run_CRD(self.opt, self.global_model)
                # run_CRD.train(self.global_model)
                # device = torch.device("cpu")
                # self.global_model = self.global_model.to(device)
                self.send_models()



    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
