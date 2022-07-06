# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import datetime
import math
import os
import sys
import time
import warnings
import collections
from copy import deepcopy

sys.path.append('./')

import misc.logger as logging
import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import data
from data import build_loader
from misc.config_ds import get_config
from misc.ds_utils import (
    calculate_slope_iterwise,
    calculate_slope_epochwise,
    get_efficiency,
    gumbel_softmax,
)
from misc.loss_ops import AdaptiveLossSoft, KLLossSoft
from misc.lr_scheduler import build_scheduler
from misc.optimizer import build_optimizer
from misc.utils import (
    load_checkpoint,
    save_checkpoint,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor,
)
from models import build_model
from models.utils_ds import (
    Changable_Act,
    Learnable_Relu_Hard,
    Learnable_Relu6_Hard,
    Learnable_Gelu_Hard,
    Final_Act
)
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from timm.utils import ModelEma
from timm.utils import accuracy, AverageMeter

try:
    from apex import amp
except ImportError:
    amp = None

from fvcore.nn import FlopCountAnalysis


logger = logging.get_logger(__name__)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--working-dir',
        type=str, required=False,
        default="manifold://ondevice_ai_tools/tree/users/yongganfu/experiments/depth_shrink",
        help='root dir for models and logs',
    )
    parser.add_argument('--data_path',
        type=str,
        default="/data1/dataset/ILSVRC/Data/CLS-LOC/",
        help='path to imagenet',
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', type=str, default="", help='resume path')
    parser.add_argument('--pretrain', type=str, default="", help='pretrain path')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed data parallel settings
    parser.add_argument("--machine-rank", default=0, type=int, help="machine rank, distributed setting")
    parser.add_argument("--num-machines", default=1, type=int, help="number of nodes, distributed setting")
    parser.add_argument("--workflow-run-id", default="", type=str, help="fblearner job id")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:10001", type=str, help="init method, distributed setting")

    parser.add_argument('--distill', action='store_true', help='If use distillation in ViT training')

    parser.add_argument('--dp', action='store_true', help='If use data parallel')

    args, unparsed = parser.parse_known_args()

    # setup the work dir
    args.output = args.working_dir

    config = get_config(args)
    return args, config


def _setup_worker_env(gpu, ngpus_per_node, config, dp=False):
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    config.defrost()
    config.RANK = gpu + config.machine_rank * ngpus_per_node # across machines
    config.WORLD_SIZE = ngpus_per_node * config.num_nodes
    torch.distributed.init_process_group(
        backend='nccl', init_method=config.dist_url, world_size=config.WORLD_SIZE, rank=config.RANK
    )
    config.gpu = gpu
    config.LOCAL_RANK = gpu
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.barrier()

    # seed = config.SEED + dist.get_rank()
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    assert dist.get_world_size() == config.WORLD_SIZE, "DDP is not properply initialized."
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / config.TRAIN.BASE_BATCH_SIZE
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / config.TRAIN.BASE_BATCH_SIZE
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / config.TRAIN.BASE_BATCH_SIZE
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # Setup logging format.
    logging.setup_logging(os.path.join(config.OUTPUT, "stdout.log"), "a")

    # backup the config
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())


def nce(t_layers, s_layers, N_c=1):
    c_loss = 0.
    proxy_criterion = SoftTargetCrossEntropy()

    idx = 0
    B, C = t_layers[idx].shape[0], t_layers[idx].shape[1]
    # t_layers[idx] = t_layers[idx].reshape(B, C, -1)
    N = N_c * N_c
    # t_layers[idx] = t_layers[idx].reshape(B, C, -1).reshape(B, -1, N_t, N_t)
    t_layers[idx] = F.adaptive_avg_pool2d(t_layers[idx], (N_c, N_c)).reshape(B, -1)
    t_layers[idx] = F.normalize(t_layers[idx], dim=-1)
    t_logits = t_layers[idx] @ t_layers[idx].permute(1, 0) - torch.diag(torch.ones_like(t_layers[idx][:, 0])).reshape(t_layers[idx].shape[0], t_layers[idx].shape[0]) * 1
    for idx in range(len(t_layers)):
        N_s = int(s_layers[idx].shape[1]**.5)
        s_layer = F.interpolate(s_layers[idx].permute(0, 2, 1).reshape(B, -1, N_s, N_s), size=(N_c, N_c)).reshape(B, -1) # .permute(0, 2, 1)
        s_layer = F.normalize(s_layer, dim=-1)
        s_logits = s_layer @ s_layer.permute(1, 0) - torch.diag(torch.ones_like(s_layer[:, 0])) * 1 # .reshape(s_layer.shape[1], s_layer.shape[1]) * 1

        c_loss += 1. * proxy_criterion(s_logits, t_logits.softmax(dim=-1))

    return c_loss / len(t_layers) * 2.


def correlation(t_layers, s_layers, N_c=14):
    c_loss = 0.
    proxy_criterion = SoftTargetCrossEntropy() #

    idx = 0
    B, C = t_layers[idx].shape[0], t_layers[idx].shape[1]
    t_layers[idx] = t_layers[idx].reshape(B, C, -1)
    N_t = int(t_layers[idx].shape[-1]**.5)
    N = N_c * N_c
    t_layers[idx] = t_layers[idx].reshape(B, C, -1).reshape(B, -1, N_t, N_t)
    t_layers[idx] = F.interpolate(t_layers[idx], size=(N_c, N_c)).reshape(B, -1, N).permute(0, 2, 1)
    t_layers[idx] = F.normalize(t_layers[idx], dim=-1)
    t_logits = t_layers[idx] @ t_layers[idx].transpose(1, 2) - torch.diag(torch.ones_like(t_layers[idx][0, :, 0])).reshape(1, t_layers[idx].shape[1], t_layers[idx].shape[1]) * 1

    for idx in range(len(t_layers)):
        N_s = int(s_layers[idx].shape[1]**.5)

        # t_layers[idx] = F.normalize(t_layers[idx], dim=-1)
        s_layer = F.interpolate(s_layers[idx].permute(0, 2, 1).reshape(B, -1, N_s, N_s), size=(N_c, N_c)).reshape(B, -1, N).permute(0, 2, 1)
        s_layer = F.normalize(s_layer, dim=-1)

        s_logits = s_layer @ s_layer.transpose(1, 2) - torch.diag(torch.ones_like(s_layer[0, :, 0])).reshape(1, s_layer.shape[1], s_layer.shape[1]) * 1

        temperature = 1.5
        c_loss += 1. * proxy_criterion(s_logits * temperature, (t_logits * temperature).softmax(dim=-1))

    return c_loss / len(t_layers) * 1.


def main_worker(gpu, ngpus_per_node, config, dp=False):
    if not dp:
        _setup_worker_env(gpu, ngpus_per_node, config)
    else:
        # Setup logging format.
        logging.setup_logging(os.path.join(config.OUTPUT, "stdout.log"), "a", dp=True)

    if 'tf_' in config.MODEL.TYPE:
        data.build.IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
        data.build.IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, dp)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if config.DS.SEARCH:
        for module in model.modules():
            if isinstance(module, Changable_Act):
                module.set_act_fun(config.DS.ACT_FUN)

    config.defrost()
    if not config.DS.SEARCH and not config.DS.KEEP_ALL_ACT:
        assert config.DS.ACT_FROM_SEARCH or config.DS.ACT_FROM_LIST
        
        if config.DS.ACT_FROM_SEARCH:
            assert config.DS.SEARCH_CKPT is not None

            with open(config.DS.SEARCH_CKPT, "rb") as fp:
                ckpt = torch.load(fp, map_location='cpu')['model']

            slope_param = []
            for key in ckpt.keys():
                if 'slope_param' in key:
                    slope_param.append(ckpt[key].data.item())

            if 'relu' in config.DS.ACT_FUN or 'gelu' in config.DS.ACT_FUN:
                all_zero = np.zeros(len(slope_param))
                rank_s2l = np.argsort(slope_param)

                act_ind_list = all_zero.copy()

                k = int(len(slope_param)*config.DS.L0_SPARSITY)
                act_ind_list[rank_s2l[:k]] = 1

                logger.info('Keep %.1f%%  activation funtions: %s', config.DS.L0_SPARSITY*100, act_ind_list)
                config.act_ind_list = list(act_ind_list)

                # input()

            else:
                logger.info('Not implemented activation fuction:%s', config.DS.ACT_FUN)

        elif config.DS.ACT_FROM_LIST:
            config.act_ind_list = config.DS.ACT_LIST
        
        else:
            logger.info('Plz indicate layerwise activation functions.')
            sys.exit()

    config.freeze()

    # if args.keep_all_act is True, all act fun will be kept; otherwise if args.decay_slope is False, then all changable slopes will be set to the end slope
    if config.EVAL_MODE or (not config.DS.SEARCH and not config.DS.DECAY_SLOPE and not config.DS.KEEP_ALL_ACT):
        slope = []
        for act_ind in config.act_ind_list:
            if act_ind:
                slope.append(config.DS.START_SLOPE)
            else:
                slope.append(config.DS.END_SLOPE)

        model.set_slope(slope)


    if config.DS.ADD_FINAL_ACT and not config.DS.SEARCH and not config.DS.PROG_REMOVE and not config.DS.KEEP_ALL_ACT:
        assert 'block' in config.MODEL.TYPE

        act_name = config.DS.ADD_FINAL_ACT
        final_act_lr_scale = config.DS.FINAL_ACT_LR_SCALE

        slope = []
        for act_ind in config.act_ind_list:
            if act_ind:
                slope.append(config.DS.START_SLOPE)
            else:
                slope.append(config.DS.END_SLOPE)

        model.add_final_act(slope, act_name, final_act_lr_scale)

    if config.DS.REMOVE_BLOCK:
        assert 'block' in config.MODEL.TYPE
        slope = []
        for act_ind in config.act_ind_list:
            if act_ind:
                slope.append(config.DS.START_SLOPE)
            else:
                slope.append(config.DS.END_SLOPE)

        model.remove_block(slope)


    if config.DS.PRETRAINED:
        checkpoint_name = config.DS.PRETRAINED

        with open(checkpoint_name, "rb") as fp:
            checkpoint = torch.load(fp, map_location='cpu')

        if 'vgg' in checkpoint_name:
            checkpoint_new = {}
            for k, v in checkpoint.items():
                k_r = k
                k_r = k_r.replace('classifier.0', 'pre_logits.fc1')
                k_r = k_r.replace('classifier.3', 'pre_logits.fc2')
                k_r = k_r.replace('classifier.6', 'head.fc')
                if 'classifier.0.weight' in k:
                    v = v.reshape(-1, 512, 7, 7)
                if 'classifier.3.weight' in k:
                    v = v.reshape(-1, 4096, 1, 1)
                checkpoint_new[k_r] = v

            checkpoint = checkpoint_new

        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        logger.info("Successfully load pretrained model: %s", checkpoint_name)

    model.cuda()
    logger.info(str(model))


    if config.EVAL_MODE:
        acc1, acc5, loss = validate(config, data_loader_val, model, dp=dp)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        return


    if config.THROUGHPUT_MODE:
        if config.DS.MERGE:
            assert 'block' in config.MODEL.TYPE
            slope = []
            for act_ind in config.act_ind_list:
                if act_ind:
                    slope.append(config.DS.START_SLOPE)
                else:
                    slope.append(config.DS.END_SLOPE)

            model.merge_block(slope)
        
        # print(model)
        
        for module in model.modules():
            if isinstance(module, Changable_Act) or isinstance(module, Final_Act):
                module.act_fun = torch.nn.ReLU6(inplace=False)
                module.slope = None
        
        inputs = (torch.randn((1,3,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)).cuda(),)
        flops = FlopCountAnalysis(model, inputs)
        logger.info(f"number of GFLOPs: {flops.total() / 1e9}")

        throughput(data_loader_val, model, logger)
        return


    if config.DS.RANDOM_DROP or config.DS.NO_BN_STATS:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None

    model_ema = ModelEma(model, decay=.99996, device='', resume='')

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    if not dp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    else:
        model = torch.nn.parallel.DataParallel(model)

    model_without_ddp = model.module


    if config.DS.DISTILL:
        teachers = build_model(config)

        checkpoint_name = config.DS.PRETRAINED 
        
        with open(checkpoint_name, "rb") as fp:
            checkpoint = torch.load(fp, map_location='cpu')

        if 'vgg' in checkpoint_name:
            checkpoint_new = {}
            for k, v in checkpoint.items():
                k_r = k
                k_r = k_r.replace('classifier.0', 'pre_logits.fc1')
                k_r = k_r.replace('classifier.3', 'pre_logits.fc2')
                k_r = k_r.replace('classifier.6', 'head.fc')
                if 'classifier.0.weight' in k:
                    v = v.reshape(-1, 512, 7, 7)
                if 'classifier.3.weight' in k:
                    v = v.reshape(-1, 4096, 1, 1)
                checkpoint_new[k_r] = v

            checkpoint = checkpoint_new

        if 'model' in checkpoint.keys():
            teachers.load_state_dict(checkpoint['model'], strict=True)
        else:
            teachers.load_state_dict(checkpoint, strict=True)

        teachers = teachers.cuda()
        teachers.eval()

        if not dp:
            teachers = torch.nn.parallel.DistributedDataParallel(teachers, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        else:
            teachers = torch.nn.parallel.DataParallel(teachers)

        logger.info("Successfully build teacher model")

    else:
        teachers = None

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params (M): {n_parameters / 1e6}")

    model_without_ddp.eval()
    inputs = (torch.randn((1,3,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)).cuda(),)
    flops = FlopCountAnalysis(model_without_ddp, inputs)
    logger.info(f"number of GFLOPs: {flops.total() / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        model_ema.ema = deepcopy(model_without_ddp)

        if config.EVAL_MODE:
            if not config.DS.SEARCH and not config.DS.KEEP_ALL_ACT:
                slope = []
                for act_ind in config.act_ind_list:
                    if act_ind:
                        slope.append(config.DS.START_SLOPE)
                    else:
                        slope.append(config.DS.END_SLOPE)

                model.module.set_slope(slope)

            acc1, acc5, loss = validate(config, data_loader_val, model, dp=dp)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            return

    config.defrost()

    config.iter_per_epoch = len(data_loader_train)
    config.lr_steps = config.TRAIN.EPOCHS * len(data_loader_train)

    if config.DS.END_EPOCH < 0:
        config.DS.END_EPOCH = config.TRAIN.EPOCHS - 1

    config.freeze()

    if config.DS.DISTILL_FEATURE and not config.DS.PROG_REMOVE:
        assert config.DS.DISTILL

        slope = []
        for act_ind in config.act_ind_list:
            if act_ind:
                slope.append(config.DS.START_SLOPE)
            else:
                slope.append(config.DS.END_SLOPE)

        model.module.set_stu_handler(config)
        teachers.module.set_tea_handler(config, slope)


    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if not dp:
            data_loader_train.sampler.set_epoch(epoch)

        if not config.DS.SEARCH and not config.DS.KEEP_ALL_ACT and config.DS.DECAY_SLOPE and config.DS.DECAY_MODE == 'epoch':
            slope = calculate_slope_epochwise(epoch, config)
            model.module.set_slope(slope)


        if config.DS.PROG_REMOVE and not config.DS.SEARCH and not config.DS.KEEP_ALL_ACT:
            rm_cnt = sum([1 if act_ind == 0 else 0 for act_ind in config.act_ind_list])

            epoch_per_rm = config.DS.PROG_REMOVE_EPOCH // (rm_cnt - 1)

            num_rm = min(epoch // epoch_per_rm + 1, rm_cnt)

            slope = []

            if config.DS.PROG_REMOVE_MODE == 'forward':
                for act_ind in config.act_ind_list:
                    if act_ind:
                        slope.append(config.DS.START_SLOPE)
                    else:
                        if num_rm > 0:
                            slope.append(config.DS.END_SLOPE)
                            num_rm = num_rm - 1
                        else:
                            slope.append(config.DS.START_SLOPE)

            else:
                num_not_rm = rm_cnt - num_rm
                for act_ind in config.act_ind_list:
                    if act_ind:
                        slope.append(config.DS.START_SLOPE)
                    else:
                        if num_not_rm > 0:
                            slope.append(config.DS.START_SLOPE)
                            num_not_rm = num_not_rm - 1
                        else:
                            slope.append(config.DS.END_SLOPE)

            model.module.set_slope(slope)

            if config.DS.ADD_FINAL_ACT:
                act_name = config.DS.ADD_FINAL_ACT
                final_act_lr_scale = config.DS.FINAL_ACT_LR_SCALE

                model.module.add_final_act(slope, act_name)

            if config.DS.DISTILL_FEATURE:
                model.module.set_stu_handler(config)
                teachers.module.set_tea_handler(config, slope)


        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, epoch, teachers, model_ema)

        if config.DS.SEARCH and config.DS.L0_SPARSITY > 0:
            assert 'learnable' in config.DS.ACT_FUN
            assert 'hard' in config.DS.ACT_FUN

            slope_param = []
            for name, param in model.named_parameters():
                if 'slope_param' in name:
                    slope_param.append(param.data.item())

            if config.DS.ACT_FUN == 'learnable_relu_hard' or config.DS.ACT_FUN == 'learnable_relu6_hard' or config.DS.ACT_FUN == 'learnable_gelu_hard':
                id_list = np.argsort(slope_param)[:int(config.DS.L0_SPARSITY*len(slope_param))]

                flag_list = np.zeros(len(slope_param))
                flag_list[id_list] = 1

                i = 0
                for module in model.modules():
                    if isinstance(module, Learnable_Relu_Hard) or isinstance(module, Learnable_Relu6_Hard) or isinstance(module, Learnable_Gelu_Hard):
                        module.set_flag(flag_list[i])
                        i += 1

            else:
                logger.info('Not implemented activation fuction:%s' % config.DS.ACT_FUN)
                sys.exit()

        acc1, acc5, loss = validate(config, data_loader_val, model, dp=dp)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.2f}%")

        if config.DS.KEEP_ALL_ACT or ((not config.DS.DECAY_SLOPE or epoch >= config.DS.END_EPOCH) and (not config.DS.PROG_REMOVE or epoch >= config.DS.PROG_REMOVE_EPOCH)):   # the test acc if valid since the slope is fixed (the same with the inference one)
            max_accuracy = max(max_accuracy, acc1)

        logger.info(f'Max accuracy (after decay): {max_accuracy:.2f}%')

        if (dp or dist.get_rank() == 0) and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


    if config.DS.SEARCH:
        slope_param = []
        for name, param in model.named_parameters():
            if 'slope_param' in name:
                slope_param.append(param.data.item())

        if config.DS.ACT_FUN == 'learnable_relu_hard' or config.DS.ACT_FUN == 'learnable_relu6_hard' or config.DS.ACT_FUN == 'learnable_gelu_hard' :
            all_zero = np.zeros(len(slope_param))
            rank_s2l = np.argsort(slope_param)

            ratio_list = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            for ratio in ratio_list:
                act_ind_list = all_zero.copy()

                k = int(len(slope_param)*ratio)
                act_ind_list[rank_s2l[:k]] = 1

                logger.info('Keep %.1f%%  activation funtions: %s', ratio*100, act_ind_list)

        else:
            logger.info('Not implemented activation fuction:%s', config.DS.ACT_FUN)


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, epoch_id, teacher=None, model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    ce_criterion = nn.CrossEntropyLoss()

    # teacher_criterion = AdaptiveLossSoft(alpha_min=-1., alpha_max=1.)
    teacher_criterion = KLLossSoft()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        data_time.update(time.time() - end)

        if not config.DS.SEARCH and not config.DS.KEEP_ALL_ACT and config.DS.DECAY_SLOPE and config.DS.DECAY_MODE == 'iter':
            if not config.DS.RANDOM_DROP:
                iters = epoch_id * len(data_loader) + idx
                slope = calculate_slope_iterwise(iters, config)
            else:
                ind_list = [np.random.randint(0, 2) for _ in range(len(config.act_ind_list))]

                slope = []
                for act_ind in ind_list:
                    if act_ind:
                        slope.append(config.DS.START_SLOPE)
                    else:
                        slope.append(config.DS.END_SLOPE)

            model.module.set_slope(slope)

        if config.DS.SEARCH and config.DS.L0_SPARSITY > 0:
            assert 'learnable' in config.DS.ACT_FUN
            assert 'hard' in config.DS.ACT_FUN

            slope_param = []
            for name, param in model.named_parameters():
                if 'slope_param' in name:
                    slope_param.append(param.data.item())

            if config.DS.ACT_FUN == 'learnable_relu_hard' or config.DS.ACT_FUN == 'learnable_relu6_hard' or config.DS.ACT_FUN == 'learnable_gelu_hard':
                if config.DS.GS_SAMPLE.ENABLE and epoch_id < config.DS.GS_SAMPLE.EPOCH:
                    slope_param = torch.tensor(slope_param).cuda()
                    temp = config.DS.GS_SAMPLE.INIT_TEMP * config.DS.GS_SAMPLE.DECAY_RATE ** epoch_id
                    slope_param = gumbel_softmax(slope_param, temperature=temp).data.cpu().numpy()

                id_list = np.argsort(slope_param)[:int(config.DS.L0_SPARSITY*len(slope_param))]

                flag_list = np.zeros(len(slope_param))
                flag_list[id_list] = 1

                i = 0
                for module in model.modules():
                    if isinstance(module, Learnable_Relu_Hard) or isinstance(module, Learnable_Relu6_Hard) or isinstance(module, Learnable_Gelu_Hard):
                        module.set_flag(flag_list[i])
                        i += 1

            else:
                logger.info('Not implemented activation fuction:%s' % config.DS.ACT_FUN)
                sys.exit()


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if teacher is None:
            pass
        else:
            with torch.no_grad():
                # soft_logits, t_layers = teacher[0](samples)
                # soft_logits1 = teacher[1](samples)
                soft_logits = teacher(samples)

        outputs = model(samples)

        # if teacher is not None:
        #     t_layers = [t_layers[0]] * len(s_layers)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if teacher is None:
                loss = criterion(outputs, targets)

            else:
                loss = teacher_criterion(outputs, soft_logits, target=targets, alpha=config.DS.DISTILL_WEIGHT)

                if config.DS.DISTILL_FEATURE:
                    assert len(model.module.stu_feature_list.keys()) == len(teacher.module.tea_feature_list.keys())

                    for device_id in model.module.stu_feature_list.keys():
                        assert len(model.module.stu_feature_list[device_id]) == len(model.module.tea_feature_list[device_id])

                    feature_loss_list = []
                    for device_id in model.module.stu_feature_list.keys():
                        feature_loss = 0
                        for layer_id in range(len(model.module.stu_feature_list[device_id])):
                            feature_loss += config.DS.DISTILL_FEATURE_WEIGHT * nn.MSELoss()(model.module.stu_feature_list[device_id][layer_id], teacher.module.tea_feature_list[device_id][layer_id])

                        feature_loss_list.append(feature_loss)

                    for feature_loss in feature_loss_list:
                        loss += feature_loss.to(loss.device) / len(feature_loss_list)

                    model.module.clear_feature_list()
                    teacher.module.clear_feature_list()

            lat_cost = 0
            for k in range(len(slope_param_list)):
                lat_cost = lat_cost + (config.DS.LAT_BEFORE[k] - config.DS.LAT_AFTER[k]) * slope_param_list[k]
            loss = loss + lat_cost * config.DS.LAT_COST_WEIGHT

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            if teacher is None:
                loss = criterion(outputs, targets)

            else:
                # ce_loss = .5 * ce_criterion(outputs[1], soft_logits.max(dim=-1)[1]) + .5 * ce_criterion(outputs[0], soft_logits1.max(dim=-1)[1])
                # ce_loss = .5 * teacher_criterion(outputs[1], soft_logits) + .5 * teacher_criterion(outputs[0], soft_logits1)
                # local_ce_loss = .5 * criterion(local_outputs[1], local_soft_logits.softmax(dim=-1)) + .5 * criterion(local_outputs[0], local_soft_logits.softmax(dim=-1))
                # loss = ce_loss # + nce(t_layers, s_layers)
                # loss = ce_loss + correlation(t_layers, s_layers)
                # loss = ce_loss + local_ce_loss * .5 + correlation(t_layers, s_layers)

                loss = teacher_criterion(outputs, soft_logits, target=targets, alpha=config.DS.DISTILL_WEIGHT)

                if config.DS.DISTILL_FEATURE:
                    assert len(model.module.stu_feature_list.keys()) == len(teacher.module.tea_feature_list.keys())

                    for device_id in model.module.stu_feature_list.keys():
                        assert len(model.module.stu_feature_list[device_id]) == len(teacher.module.tea_feature_list[device_id])

                    feature_loss_list = []
                    for device_id in model.module.stu_feature_list.keys():
                        feature_loss = 0
                        for layer_id in range(len(model.module.stu_feature_list[device_id])):
                            feature_loss += config.DS.DISTILL_FEATURE_WEIGHT * nn.MSELoss()(model.module.stu_feature_list[device_id][layer_id], teacher.module.tea_feature_list[device_id][layer_id])

                        feature_loss_list.append(feature_loss)

                    for feature_loss in feature_loss_list:
                        loss += feature_loss.to(loss.device) / len(feature_loss_list)

                    model.module.clear_feature_list()
                    teacher.module.clear_feature_list()
            
            slope_param_list = []
            for name, module in model.named_modules():
                if hasattr(module, 'slope_param'):
                    slope_param_list.append(module.slope_param)

            lat_cost = 0
            for k in range(len(slope_param_list)):
                lat_cost = lat_cost + (config.DS.LAT_BEFORE[k] - config.DS.LAT_AFTER[k]) * slope_param_list[k]
            loss = loss + lat_cost * config.DS.LAT_COST_WEIGHT

            if not math.isfinite(loss.item()):
                continue
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'batch {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    if hasattr(model.module, 'slope'):
        logger.info('Current slope: %s \t', model.module.slope)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")



@torch.no_grad()
def validate(config, data_loader, model, dp=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output.detach(), target)
        acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))

        if not dp:
            acc1 = reduce_tensor(acc1).item()
            acc5 = reduce_tensor(acc5).item()
            loss = reduce_tensor(loss).item()

            batch_size = images.size(0)
            corr1, corr5, loss = acc1 * batch_size, acc5 * batch_size, loss * batch_size
            stats = torch.tensor([corr1, corr5, loss, batch_size], device=config.LOCAL_RANK)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1 / batch_size, corr5 / batch_size, loss / batch_size

            loss_meter.update(loss, target.size(0))
            acc1_meter.update(acc1, target.size(0))
            acc5_meter.update(acc5, target.size(0))
        
        else:
            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return



if __name__ == '__main__':
    args, config = parse_option()

    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                   'This will turn on the CUDNN deterministic setting, '
                   'which can slow down your training considerably! '
                   'You may see unexpected behavior when restarting '
                   'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()

    if not args.dp:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, False)
        )

    else:
        config.defrost()
        config.PRINT_FREQ = 1
        config.freeze()
        main_worker(gpu=None, ngpus_per_node=ngpus_per_node, config=config, dp=True)
