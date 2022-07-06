# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def get_act_fun(name):
    def relu(x, slope=0):
        return F.leaky_relu(x, negative_slope=slope)

    def relu6(x, slope=0):
        return F.leaky_relu(x, negative_slope=slope) + (slope-1) * F.relu(x-6)

    def gelu(x, slope=0):  # slope decay not implemented
        return F.gelu(x)

    def leaky_relu(x, slope=0.01):
        return F.leaky_relu(x, negative_slope=slope)

    def swish(x, slope=1):
        x = x / (1 + slope * torch.exp(-x))
        return x

    def swish_old(x, slope=1):
        x = 2 * x / (1 + torch.exp(-slope*x))
        return x

    def identity(x):
        return torch.nn.Identity()(x)

    if name == 'relu':
        return relu

    elif name == 'relu6':
        return relu6

    elif name == 'leaky_relu':
        return leaky_relu

    elif name == 'swish':
        return swish

    elif name == 'swish_old':
        return swish_old

    elif name == 'gelu':
        return gelu

    elif name == 'identity':
        return identity

    else:
        print('No such activation func:', name)
        sys.exit()


class Changable_Act(nn.Module):
    def __init__(self, name='relu', slope=None, inplace=False):
        super(Changable_Act, self).__init__()

        self.name = name
        self.slope = slope

        if self.name == 'learnable_relu':
            self.act_fun = Learnable_Relu()
        elif self.name == 'learnable_relu6':
            self.act_fun = Learnable_Relu6()
        elif self.name == 'learnable_relu_hard':
            self.act_fun = Learnable_Relu_Hard()
        elif self.name == 'learnable_relu6_hard':
            self.act_fun = Learnable_Relu6_Hard()
        elif self.name == 'learnable_gelu_hard':
            self.act_fun = Learnable_Gelu_Hard()
        else:
            self.act_fun = get_act_fun(self.name)


    def set_slope(self, slope):
        self.slope = slope


    def set_act_fun(self, name):
        self.name = name

        if self.name == 'learnable_relu':
            self.act_fun = Learnable_Relu()
        elif self.name == 'learnable_relu6':
            self.act_fun = Learnable_Relu6()
        elif self.name == 'learnable_relu_hard':
            self.act_fun = Learnable_Relu_Hard()
        elif self.name == 'learnable_relu6_hard':
            self.act_fun = Learnable_Relu6_Hard()
        elif self.name == 'learnable_gelu_hard':
            self.act_fun = Learnable_Gelu_Hard()
        else:
            self.act_fun = get_act_fun(self.name)


    def forward(self, x):
        if 'learnable' in self.name:
            return self.act_fun(x)

        else:
            if self.slope is None:
                return self.act_fun(x)
            else:
                return self.act_fun(x, self.slope)


def decorator(name):
    def fun(inplace=False):
        return Changable_Act(name=name, inplace=inplace)

    return fun


Changable_Swish = decorator(name='swish')

Changable_Relu = decorator(name='relu')

Changable_Relu6 = decorator(name='relu6')

Changable_Gelu = decorator(name='gelu')


class Learnable_Relu(nn.Module):
    def __init__(self, slope_init=0.):
        super(Learnable_Relu, self).__init__()

        self.slope = nn.Parameter(torch.tensor(slope_init))

        self.slope_lr_scale = 1

    def forward(self, x):
        slope = (self.slope - self.slope * self.slope_lr_scale).detach() + self.slope * self.slope_lr_scale

        x = F.relu(x) + (x - F.relu(x)) * torch.clamp(slope, 0, 1)

        return x


class Learnable_Relu6(nn.Module):
    def __init__(self, slope_init=0.):
        super(Learnable_Relu6, self).__init__()

        self.slope_param = nn.Parameter(torch.tensor(slope_init))

        self.slope_lr_scale = 1

    def forward(self, x):
        slope = (self.slope_param - self.slope_param * self.slope_lr_scale).detach() + self.slope_param * self.slope_lr_scale

        x = F.relu(x) + (x - F.relu(x)) * torch.clamp(slope, 0, 1) + (torch.clamp(slope, 0, 1)-1) * F.relu(x-6)

        return x



class Learnable_Relu_Hard(nn.Module):
    def __init__(self, slope_init=0.):
        super(Learnable_Relu_Hard, self).__init__()

        self.slope_param = nn.Parameter(torch.tensor(slope_init))

        self.flag = 1

        self.slope_lr_scale = 1

    def set_flag(self, flag):
        self.flag = flag


    def forward(self, x):
        slope = (self.slope_param - self.slope_param * self.slope_lr_scale).detach() + self.slope_param * self.slope_lr_scale

        if self.flag:
            x = F.relu(x) + (x - F.relu(x)) * (0 - torch.clamp(slope, 0, 1).detach() + torch.clamp(slope, 0, 1))

        else:
            x = F.relu(x) + (x - F.relu(x)) * (1 - torch.clamp(slope, 0, 1).detach() + torch.clamp(slope, 0, 1))

        return x


class Learnable_Relu6_Hard(nn.Module):
    def __init__(self, slope_init=0.):
        super(Learnable_Relu6_Hard, self).__init__()

        self.slope_param = nn.Parameter(torch.tensor(slope_init))

        self.flag = 1

        self.slope_lr_scale = 1

    def set_flag(self, flag):
        self.flag = flag


    def forward(self, x):
        x_act = F.relu(x) - F.relu(x-6)

        slope = (self.slope_param - self.slope_param * self.slope_lr_scale).detach() + self.slope_param * self.slope_lr_scale

        if self.flag:
            x = x_act + (x - x_act) * (0 - torch.clamp(slope, 0, 1).detach() + torch.clamp(slope, 0, 1))

        else:
            x = x_act + (x - x_act) * (1 - torch.clamp(slope, 0, 1).detach() + torch.clamp(slope, 0, 1))

        return x


class Learnable_Gelu_Hard(nn.Module):
    def __init__(self, slope_init=0.):
        super(Learnable_Gelu_Hard, self).__init__()

        self.slope_param = nn.Parameter(torch.tensor(slope_init))

        self.flag = 1

        self.slope_lr_scale = 1

    def set_flag(self, flag):
        self.flag = flag


    def forward(self, x):
        x_act = F.gelu(x)

        slope = (self.slope_param - self.slope_param * self.slope_lr_scale).detach() + self.slope_param * self.slope_lr_scale

        if self.flag:
            x = x_act + (x - x_act) * (0 - torch.clamp(slope, 0, 1).detach() + torch.clamp(slope, 0, 1))

        else:
            x = x_act + (x - x_act) * (1 - torch.clamp(slope, 0, 1).detach() + torch.clamp(slope, 0, 1))

        return x



class Final_Act(nn.Module):
    def __init__(self, name='relu', final_act_lr_scale=1):
        super(Final_Act, self).__init__()

        self.name = name

        if self.name == 'learnable_relu':
            self.act_fun = Learnable_Relu(slope_init=1.)
        elif self.name == 'learnable_relu6':
            self.act_fun = Learnable_Relu6(slope_init=1.)
        elif self.name == 'learnable_relu_hard':
            self.act_fun = Learnable_Relu_Hard(slope_init=1.)
        elif self.name == 'learnable_relu6_hard':
            self.act_fun = Learnable_Relu6_Hard(slope_init=1.)
        elif self.name == 'learnable_gelu_hard':
            self.act_fun = Learnable_Gelu_Hard(slope_init=1.)
        else:
            self.act_fun = get_act_fun(self.name)

        if 'learnable' in name:
            self.act_fun.slope_lr_scale = final_act_lr_scale

    def forward(self, x):
        return self.act_fun(x)
