import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.autograd import Variable
# from utils.data_utils import read_client_data
from baseModel import BaseModel




def from_one_hot(data):
    data_new = []
    for p in data:
        for i in range(len(p)):
            if p[i] == 1:
                data_new.append(i)
    return torch.tensor(data_new).long()

# def from_one_hot(data):
#     data_new = []
#     for p in data:
#         data = []
#         for i in range(len(p)):
#             if p[i] == 1:
#                 data.append(i)
#         data_new.append(data)
#     return torch.tensor(data_new).long()

def get_input(train_data, eval_data):
    train_data = {k: np.array(v) for k, v in train_data.items()}
    train_data_x = torch.tensor(train_data['x']).to(torch.float32)
    # train_data_x = torch.reshape(train_data_x, shape=[-1, 3, 32, 32])
    train_data_y = from_one_hot((train_data['y']))
    # train_data_y = torch.tensor(train_data['y']).long()
    # print(train_data_y.size())

    eval_data = {k: np.array(v) for k, v in eval_data.items()}
    eval_data_x = torch.tensor(eval_data['x']).to(torch.float32)
    # eval_data_x = torch.reshape(eval_data_x, shape=[-1, 3, 32, 32])
    eval_data_y = from_one_hot((eval_data['y']))
    # eval_data_y = torch.tensor(eval_data['y']).long()
    return  train_data_x, train_data_y, eval_data_x, eval_data_y

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self,id, train_samples, test_samples, args,model=None):
        self.model = copy.deepcopy(model)
        self.k = 0
        self.alpha = args['alpha']
        self.rho = args['rho']
        self.w_i = args['w_i']
        self.mu_i = args['seed']
        self.theta_kp1 = copy.deepcopy(self.model)
        self.num_train = len(train_samples['y'])
        self.num_test = len(test_samples['y'])
        self.num_samples = self.num_train + self.num_test
        # self.algorithm = args.algorithm
        # self.dataset = args.dataset
        # self.device = args.device
        self.yy_k = self.construct_yy_k()
        self.delta = Variable(torch.tensor(1000.0), requires_grad=False)

        self.id = id  # integer
        # self.save_folder_name = args.save_folder_name
        #
        # self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        # self.batch_size = args.batch_size
        # # self.learning_rate = args.local_learning_rate
        # self.local_epochs = args.local_epochs

        # check BatchNorm
        # self.has_BatchNorm = False
        # for layer in self.model.children():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         self.has_BatchNorm = True
        #         break
        #
        # self.train_slow = kwargs['train_slow']
        # self.send_slow = kwargs['send_slow']
        # self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        # self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        #
        # self.privacy = args.privacy
        # self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )
        # self.learning_rate_decay = args.learning_rate_decay


    # def load_train_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     train_data = read_client_data(self.dataset, self.id, is_train=True)
    #     return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    #
    # def load_test_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     test_data = read_client_data(self.dataset, self.id, is_train=False)
    #     return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        # print(self.model.state_dict()['conv_1.weight'])
        # self.theta_kp1 = copy.deepcopy(model)

    # def clone_model(self, model, target):
    #     for param, target_param in zip(model.parameters(), target.parameters()):
    #         target_param.data = param.data.clone()
    #         # target_param.grad = param.grad.clone()
    #
    # def update_parameters(self, model, new_params):
    #     for param, new_param in zip(model.parameters(), new_params):
    #         param.data = new_param.data.clone()

    def truncated_normal_(self, tensor, mean=0, std=0.01):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def construct_yy_k(self):
            # tf.set_random_seed(123)
        # yyk = [tf.Variable(tf.truncated_normal(x.shape, stddev=0.01), name='yyk_' + x.name.split(':', 1)[0],
        #     dtype=tf.float32, trainable=False) for x in tv]
        #yyk = [Variable(self.truncated_normal_(x).to(torch.float32), requires_grad=False) for x in self.parameters()]
        yyk = [Variable(torch.zeros(x.size()), requires_grad=False) for x in self.model.parameters()]
        # yyk = [Variable(torch.normal(mean=0., std=0.01, size=x.size()), requires_grad=False) for x in self.model.parameters()]
        # yyk = [Variable(torch.nn.init.normal_(x, 0.0, 0.1).to(torch.float32), requires_grad=False) for x in self.model.parameters()]
        return yyk

    def train_avg(self, local_epoch):
        train_data_x, train_data_y, eval_data_x, eval_data_y = get_input(self.train_samples, self.test_samples)

        for i in range(local_epoch):
            self.model.train()
            pred_1 = self.model(train_data_x)
            train_loss = self.loss(pred_1, train_data_y)
            self.model.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        # pred_2 = self.model(eval_data_x)
        pred = self. model(train_data_x)
        loss = self. loss(pred, train_data_y)
        pred_train = torch.argmax(pred, dim=1)
        train_acc = torch.mean(
            torch.eq(train_data_y.type(torch.FloatTensor),
                     pred_train.type(torch.FloatTensor)).type(torch.FloatTensor))
        return train_acc, loss, self.num_samples

    def train_maml(self, local_epoch):
        train_data_x, train_data_y, eval_data_x, eval_data_y = get_input(self.train_samples, self.test_samples)

        for i in range(local_epoch):
            self.model.train()
            temp_model = copy.deepcopy(list(self.model.parameters()))

            # step 1
            output = self.model(train_data_x)
            loss = self.loss(output, train_data_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step 2
            self.optimizer.zero_grad()
            output_test = self.model(eval_data_x)
            pred_test = torch.argmax(output_test, dim=1)
            loss_test = self.loss(output_test, eval_data_y)
            loss_test.backward()

        # test_acc = torch.mean(
        #     torch.eq(train_data_y.type(torch.FloatTensor),
        #              pred_test.type(torch.FloatTensor)).type(torch.FloatTensor))

            # restore the model parameters to the one before first update
            for old_param, new_param in zip(self.model.parameters(), temp_model):
                old_param.data = new_param.data.clone()


            self.optimizer.step()
        output_final = self.model(eval_data_x)
        pred_final = torch.argmax(output_final, dim=1)
        loss_final = self.loss(output_final, eval_data_y)


        test_acc = torch.mean(torch.eq(eval_data_y.type(torch.FloatTensor),pred_final.type(torch.FloatTensor)).type(torch.FloatTensor))
        return test_acc, loss_final, self.num_samples

    def train(self):
        train_data_x, train_data_y, eval_data_x, eval_data_y = get_input(self.train_samples, self.test_samples)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # train_data_x = train_data_x.to(device)
        # train_data_y = train_data_y.to(device)
        # eval_data_x = eval_data_x.to(device)
        # eval_data_y = eval_data_y.to(device)
        # device = torch.device("cpu")
        # self.model = self.model.to(device)

        self.k += 1
        self.delta = Variable(torch.tensor(1.0 / (self.k * 10 + 100)), requires_grad=False)
        self.model.train()
        model_backup = copy.deepcopy(self.model)
        self.theta_kp1 = copy.deepcopy(self.model)
        #  第一次更新
        # for n,p in self.model.named_parameters():
        #     print(n)
        #     print(p)
        pred_1 = self.model(train_data_x)
        train_loss_1 = self.loss(pred_1, train_data_y)
        # print(pred_1)
        # print(train_loss_1)

        grads_pred_1 = []
        # self.model.zero_grad()
        train_loss_1.backward(retain_graph=True)
        # grads_pred_1.append([p.grad.detach().clone() for p in self.model.parameters()])
        for p in self.model.parameters():
            grads_pred_1.append(p.grad.clone())

        for param in self.model.parameters():
            param.grad.detach_()
            param.grad.zero_()
        # self.optimizer.step()

        for group, grads in zip(self.model.parameters(), grads_pred_1):
            group.data = group.data - self.alpha * grads

        # for p in self.model.parameters():
        #     print(p)
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        # 求meta model
        pred_2 = self.model(eval_data_x)
        train_loss_2 = self.loss(pred_2, eval_data_y)
        # print(train_loss_2)
        # print(train_loss_2)
        grads_pred_2 = []
        # self.model.zero_grad()
        train_loss_2.backward(retain_graph=True)
        for p in self.model.parameters():
            grads_pred_2.append(p.grad.clone())

        for param in self.model.parameters():
            param.grad.detach_()
            param.grad.zero_()

        # self.optimizer.step()


        # inner 梯度1
        self.set_parameters(model_backup)

        for weight, phy in zip(self.model.parameters(), grads_pred_2):
            weight.data = (weight.data + self.delta * phy)

        pred_inner_1 = self.model(train_data_x)
        loss_inner_1 = self.loss(pred_inner_1, train_data_y)
        grad_inner_1 = []
        loss_inner_1.backward(retain_graph=True)
        for p in self.model.parameters():
            grad_inner_1.append(p.grad.clone())

        for param in self.model.parameters():
            param.grad.detach_()
            param.grad.zero_()


        #inner 梯度2
        self.set_parameters(model_backup)

        for weight, phy in zip(self.model.parameters(), grads_pred_2):
            weight.data = (weight.data - self.delta * phy)

        pred_inner_2 = self.model(train_data_x)
        loss_inner_2 = self.loss(pred_inner_2, train_data_y)
        grad_inner_2 = []
        loss_inner_2.backward()
        for p in self.model.parameters():
            grad_inner_2.append(p.grad.clone())

        for param in self.model.parameters():
            param.grad.detach_()
            param.grad.zero_()
        # self.optimizer.step()


        g_kp1 = [(g1 - g2) / (2 * self.delta) for g1, g2 in zip(grad_inner_1, grad_inner_2)]

        self.set_parameters(model_backup)
        for weight, yy, grad_p, g_1_2 in zip(self.model.parameters(), self.yy_k, grads_pred_2, g_kp1):
            weight.data = weight.data - (yy + self.w_i * (grad_p - self.alpha * g_1_2)) / self.rho

        # pred_final = self.model(train_data_x)
        # loss_final = self.loss(pred_final, train_data_y)
        # print(loss_final)

        # for p in self.model.parameters():
        #     p.grad = None

        self.update_yyk()



        pred_test = torch.argmax(pred_2, dim=1)
        test_acc = torch.mean(
            torch.eq(eval_data_y.type(torch.FloatTensor),
                     pred_test.type(torch.FloatTensor)).type(torch.FloatTensor))

        train_acc = torch.mean(
            torch.eq(train_data_y.type(torch.FloatTensor),
                     torch.argmax(input=pred_1, dim=1).type(torch.FloatTensor)).type(torch.FloatTensor))
        # device = torch.device("cuda")
        # self.model = self.model.to(device)

        return train_acc.detach().numpy(), test_acc.detach().numpy(), train_loss_2.detach().numpy(), self.num_samples



    def update_yyk(self):
        yyk_kp1s = []
        yy_ks = copy.deepcopy(self.yy_k)
        for yy_k, theta_kpi, theta_kp1 in zip(yy_ks, self.model.parameters(), self.theta_kp1.parameters()):
            yyk_kp1s.append(yy_k + self.rho * (theta_kpi.data - theta_kp1.data))
        self.set_yyk(yyk_kp1s)

    def set_yyk(self, vals):
        self.yy_k = copy.deepcopy(vals)

    def fast_adapt(self):
        train_data_x, train_data_y, eval_data_x, eval_data_y = get_input(self.train_samples, self.test_samples)
        pred_1 = self.model(train_data_x)
        loss = self.loss(pred_1, train_data_y)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        pred_test = self.model(eval_data_x)
        pred_test = torch.argmax(pred_test, dim=1)
        test_acc = torch.mean(
            torch.eq(eval_data_y.type(torch.FloatTensor),
                     pred_test.type(torch.FloatTensor)).type(torch.FloatTensor))
        for p in self.model.parameters():
            p.grad = None
        return test_acc, self.num_test

    def test_avg(self):
        train_data_x, train_data_y, eval_data_x, eval_data_y = get_input(self.train_samples, self.test_samples)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(train_data_x)
            pred_test = torch.argmax(pred, dim=1)
            test_acc = torch.mean(
                torch.eq(train_data_y.type(torch.FloatTensor),
                         pred_test.type(torch.FloatTensor)).type(torch.FloatTensor))
            print(test_acc)
        return test_acc, self.num_train




    # def test_metrics(self):
    #     testloaderfull = self.load_test_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #
    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
    #
    #     with torch.no_grad():
    #         for x, y in testloaderfull:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #
    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]
    #
    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = self.num_classes
    #             if self.num_classes == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if self.num_classes == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)
    #
    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #
    #     return test_acc, test_num, auc
    #
    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #
    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     return losses, train_num
    #
    # # def get_next_train_batch(self):
    # #     try:
    # #         # Samples a new batch for persionalizing
    # #         (x, y) = next(self.iter_trainloader)
    # #     except StopIteration:
    # #         # restart the generator if the previous generator is exhausted.
    # #         self.iter_trainloader = iter(self.trainloader)
    # #         (x, y) = next(self.iter_trainloader)
    #
    # #     if type(x) == type([]):
    # #         x = x[0]
    # #     x = x.to(self.device)
    # #     y = y.to(self.device)
    #
    # #     return x, y
    #
    #
    # def save_item(self, item, item_name, item_path=None):
    #     if item_path == None:
    #         item_path = self.save_folder_name
    #     if not os.path.exists(item_path):
    #         os.makedirs(item_path)
    #     torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    #
    # def load_item(self, item_name, item_path=None):
    #     if item_path == None:
    #         item_path = self.save_folder_name
    #     return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    #
    # # @staticmethod
    # # def model_exists():
    # #     return os.path.exists(os.path.join("models", "server" + ".pt"))
