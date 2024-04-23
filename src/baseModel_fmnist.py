import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch import autograd


def active_func(x, leak=0.2):
    # return torch.maximum(x, leak * x)
    return torch.nn.ELU(x)


def loss_func(logits, label):
    losses =F.cross_entropy(logits, label)
    return torch.mean(losses)

def init_weights(layer):
    if (type(layer) == nn.Conv2d) & (type(layer) == nn.Linear):
        nn.init.normal_(layer.weight,mean=0,std=0.1)
        nn.init.constant_(layer.bias, 0.1)

class BaseModel(nn.Module):
    def __init__(self):
        # print('@BaseModel line 17 test init')
        super(BaseModel, self).__init__()

        # self.elu = nn.ELU(alpha=1.5)
        self.active_func = torch.nn.ELU()
        #self.optimize()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same', bias=True)


        # self.max_1 = self.conv_2 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=True),
        # )
        self.pool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)

        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same', bias=True)
        self.pool_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=True)
        # self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=True)
        # self.pool_3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=True)
        # self.conv_4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=True)
        #
        # self.avgpool = nn.AvgPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=True)

        self.linear = nn.Linear(3136, 10)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                # nn.init.trunc_normal_(m.weight, std=0.1, a=-0.2,b=0.2)
                nn.init.trunc_normal_(m.weight, std=0.1, a=-0.2, b=0.2)
                nn.init.constant_(m.bias, val=0.1)

        # self.conv_1.apply(init_weights)
        # self.conv_2.apply(init_weights)
        # self.conv_3.apply(init_weights)
        # self.linear.apply(init_weights)
        # self.sess = tf.Session(graph=self.graph)
            # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "TC174611125:32005")
        #self.weights = self.construct_weights()  # weights is a list


    def forward(self, input):
        x = self.conv_1(input)
        x = self.active_func(x)
        # x = self.elu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.active_func(x)
        # x = self.elu(x)
        x = self.pool_2(x)
        # x = self.conv_3(x)
        # x = self.active_func(x)
        # x = self.pool_3(x)
        # x = self.conv_4(x)
        # x = self.active_func(x)
        # # x = self.elu(x)
        # x = self.avgpool(x)

        h_pool_shape = x.size()
        h = h_pool_shape[1]
        w = h_pool_shape[2]
        c = h_pool_shape[3]
        flatten = x.reshape(-1, h * w * c)
        # print(f'action 1{flatten} ')
        # x = x.reshape(x.shape[0], -1)
        x = self.linear(flatten)
        return x