# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def calculate_slope_iterwise(iters, config):
    start_iters = config.DS.START_EPOCH * config.iter_per_epoch
    if not config.DS.DECAY_SLOPE or iters < start_iters:
        slope = [config.DS.START_SLOPE for _ in config.act_ind_list]
    else:
        if config.DS.END_EPOCH >= 0:
            assert config.DS.END_EPOCH >= config.DS.START_EPOCH

            if config.DS.END_EPOCH == config.DS.START_EPOCH:
                current_slope = config.DS.END_SLOPE

            else:
                end_iters = config.DS.END_EPOCH * config.iter_per_epoch
                k = (config.DS.END_SLOPE - config.DS.START_SLOPE) / (end_iters - start_iters)

                current_slope = config.DS.START_SLOPE + k * min(iters - start_iters, end_iters - start_iters)

        else:
            k = (config.DS.END_SLOPE - config.DS.START_SLOPE) / (config.lr_steps - start_iters)
            current_slope = config.DS.START_SLOPE + k * (iters - start_iters)

        slope = []
        for act_ind in config.act_ind_list:
            if act_ind:  # indicate the activation here will be kept after training
                slope.append(config.DS.START_SLOPE)
            else:
                slope.append(current_slope)

    return slope


def calculate_slope_epochwise(epoch, config):
    if not config.DS.DECAY_SLOPE or epoch < config.DS.START_EPOCH:
        slope = [config.DS.START_SLOPE for _ in config.act_ind_list]
    else:
        if config.DS.END_EPOCH >= 0:
            assert config.DS.END_EPOCH >= config.DS.START_EPOCH

            if config.DS.END_EPOCH == config.DS.START_EPOCH:
                current_slope = config.DS.END_SLOPE

            else:
                k = (config.DS.END_SLOPE - config.DS.START_SLOPE) / (config.DS.END_EPOCH - config.DS.START_EPOCH)

                current_slope = config.DS.START_SLOPE + k * min(epoch - config.DS.START_EPOCH, config.DS.END_EPOCH - config.DS.START_EPOCH)

        else:
            k = (config.DS.END_SLOPE - config.DS.START_SLOPE) / (config.TRAIN.EPOCHS - config.DS.START_EPOCH)
            current_slope = config.DS.START_SLOPE + k * (epoch - config.DS.START_EPOCH)

        slope = []
        for act_ind in config.act_ind_list:
            if act_ind:  # indicate the activation here will be kept after training
                slope.append(config.DS.START_SLOPE)
            else:
                slope.append(current_slope)

    return slope


def get_efficiency(act_list, network):
    if "resnet" in network:
        if "resnet50" in network:
            for i in range(len(act_list)):
                if i % 3 == 1 and act_list[i] == 1:
                    act_list[i+2] = 1
                if i % 3 == 2 and act_list[i] == 1:
                    act_list[i+1] = 1
        else:
            for i in range(len(act_list)):
                if i % 2 == 1 and act_list[i] == 1:
                    act_list[i+1] = 1
    elif "efficientnet" in network:
        pass
    return act_list, sum(act_list)



def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits, dim=-1), temperature)

    return y
