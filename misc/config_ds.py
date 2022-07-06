# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# --------------------------------------------------------
# Modified from Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
# pathmgr = PathManager()
# pathmgr.register_handler(
#     ManifoldPathHandler(num_retries=3, timeout_sec=60), allow_override=True
# )

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 6 # 24
# Prefetch factor
_C.DATA.PREFETCH_FACTOR = 6 # 12

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'resnet50_ds'
# Model name
_C.MODEL.NAME = 'resnet50_ds'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.
# Drop block rate
_C.MODEL.DROP_BLOCK_RATE = 0.
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1


_C.DS = CN()
_C.DS.ACT_FUN = 'relu'
_C.DS.KEEP_ALL_ACT = False

_C.DS.PRETRAINED = ''

_C.DS.ACT_FROM_SEARCH = False
_C.DS.SEARCH_CKPT = ''

_C.DS.ACT_FROM_LIST = False
_C.DS.ACT_LIST = [1, 1, 1, 1, 1]
_C.DS.ACT_DISTRIB = 'first'

_C.DS.DECAY_SLOPE = False
_C.DS.DECAY_MODE = 'iter'
_C.DS.START_EPOCH = 0
_C.DS.END_EPOCH = -1

_C.DS.START_SLOPE = 0
_C.DS.END_SLOPE = 1

_C.DS.SEARCH = False
_C.DS.L1_WEIGHT = 0.

_C.DS.L0_SPARSITY = 0.

_C.DS.RANDOM_DROP = False

_C.DS.NO_BN_STATS = False

_C.DS.EXPAND_RATIO = 6

_C.DS.EA = CN()
_C.DS.EA.SEARCH = False
_C.DS.EA.SPARSE_RATIO = 0.5
_C.DS.EA.POP_SIZE = 64
_C.DS.EA.SAMPLE_SIZE = 16
_C.DS.EA.CYCLES = 5000

_C.DS.GS_SAMPLE = CN()
_C.DS.GS_SAMPLE.ENABLE = False
_C.DS.GS_SAMPLE.EPOCH = 60
_C.DS.GS_SAMPLE.INIT_TEMP = 3
_C.DS.GS_SAMPLE.DECAY_RATE = 0.95

_C.DS.MERGE = False

_C.DS.ADD_FINAL_ACT = ''
_C.DS.FINAL_ACT_LR_SCALE = 1.

_C.DS.REMOVE_BLOCK = False

_C.DS.PROG_REMOVE = False
_C.DS.PROG_REMOVE_EPOCH = 120
_C.DS.PROG_REMOVE_MODE = 'forward'  # 'backward'

_C.DS.DISTILL = False
_C.DS.DISTILL_WEIGHT = 0.9
_C.DS.DISTILL_FEATURE = False
_C.DS.DISTILL_FEATURE_WEIGHT = 1.

_C.DS.LAT_BEFORE = []
_C.DS.LAT_AFTER = []
_C.DS.LAT_COST_WEIGHT = 1e-4

# # Swin Transformer parameters
# _C.MODEL.SWIN = CN()
# _C.MODEL.SWIN.PATCH_SIZE = 4
# _C.MODEL.SWIN.IN_CHANS = 3
# _C.MODEL.SWIN.EMBED_DIM = 96
# _C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
# _C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
# _C.MODEL.SWIN.WINDOW_SIZE = 7
# _C.MODEL.SWIN.MLP_RATIO = 4.
# _C.MODEL.SWIN.QKV_BIAS = True
# _C.MODEL.SWIN.QK_SCALE = None
# _C.MODEL.SWIN.APE = False
# _C.MODEL.SWIN.PATCH_NORM = True

# _C.MODEL.VIT = CN()
# _C.MODEL.VIT.PRETRAINED = False
# _C.MODEL.VIT.PRETRAINED_STRICT = False
# _C.MODEL.VIT.ANCHOR_LIST = [[1,1], [1,3], [3,1], [3,3], [5,5]]
# _C.MODEL.VIT.PATCH_EMBED_ACT_LAYER = 'hswish'
# _C.MODEL.VIT.RP_CHANNEL = [16,32,32]
# _C.MODEL.VIT.SPARSE = False

# _C.MODEL.VIT.PATCH_EMBED_TYPE = 'normal'   # ['normal', 'reshape', 'roi_align']

# _C.MODEL.VIT.POOLING_SIZE = 16

# _C.MODEL.VIT.STE_FUN = 'softmax'
# _C.MODEL.VIT.TEMP_INIT = 5
# _C.MODEL.VIT.TEMP_DECAY = 0.98

# _C.MODEL.VIT.FIX_VIT = False
# _C.MODEL.VIT.FIX_PPN = False

# _C.MODEL.VIT.PATCH_EMBED_CHANNEL = [196]
# _C.MODEL.VIT.PATCH_EMBED_KS = [1]
# _C.MODEL.VIT.PATCH_EMBED_STRIDE = [1]



# # HRNet parameters
# _C.MODEL.HRNET = CN()
# # _C.MODEL.HRNET.BRANCH_SETTINGS = [
# #                     [['mix'], [1], [24]],
# #                     [['mix','mix'], [2, 2], [18, 36]],
# #                     [['mix','mix','mix'], [2, 2, 3], [18, 36, 72]],
# #                     [['mix','mix','mix','mix'], [2, 2, 3, 4], [18, 36, 72, 144]],
# #                     [['mix','mix','mix','mix'], [2, 2, 3, 4], [18, 36, 72, 144]]
# #                 ] # [[branch type], [num block in each branch], [channel_dim]]
# _C.MODEL.HRNET.BRANCH_SETTINGS = [
#                     [['mix'], [1], [24], [None]],
#                     [['mix','swin'], [2, 2], [18, 36], [None, 3]],
#                     [['mix','swin','swin'], [2, 2, 2], [18, 36, 72], [None, 3, 6]],
#                     [['mix','swin','swin','swin'], [2, 2, 2, 2], [18, 36, 72, 144], [None, 3, 6, 8]],
#                     [['mix','swin','swin','swin'], [2, 2, 2, 2], [18, 36, 72, 144], [None, 3, 6, 8]]
#                 ] # [[branch type], [num block in each branch], [channel_dim], [num heads for vit]]
# _C.MODEL.HRNET.FUSE_BLOCK = 'mix'
# _C.MODEL.HRNET.HEAD_BLOCK = 'inv' # ['inv','inv','inv','inv']
# _C.MODEL.HRNET.HEAD_CHANNELS = [36, 72, 144, 320]
# _C.MODEL.HRNET.EXPANSION_RATIO = 4
# _C.MODEL.HRNET.KERNEL_SIZES = [3, 5, 7]
# _C.MODEL.HRNET.INPUT_CHANNEL = [16, 16] # [24,24]
# _C.MODEL.HRNET.LAST_CHANNEL = 1280 # 1600
# _C.MODEL.HRNET.INPUT_STRIDE = 4
# _C.MODEL.HRNET.ROUND_NEAREST = 2
# _C.MODEL.HRNET.WIDTH_MULT = 1.0
# _C.MODEL.HRNET.ACTIVE_FN = 'nn.ReLU'
# _C.MODEL.HRNET.CONCAT_HEAD_FOR_CLS = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 90 # 300
_C.TRAIN.WARMUP_EPOCHS = 0 # 20
_C.TRAIN.WEIGHT_DECAY = 1e-4 # 4e-5
_C.TRAIN.BASE_LR = 0.1
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.USE_CONV_PROJ = True
_C.TRAIN.MAX = 192
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'step' # 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.BASE_BATCH_SIZE = 256

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'sgd'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'none' # 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0. # 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 0. # 1.
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 50
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

_C.DISTILL = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    
    config.merge_from_file(cfg_file)

    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size

    if args.data_path:
       config.DATA.DATA_PATH = args.data_path
    #if args.zip:
    #    config.DATA.ZIP_MODE = True
    #if args.cache_mode:
    #    config.DATA.CACHE_MODE = args.cache_mode

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.pretrain:
        config.DS.PRETRAINED = args.pretrain
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.distill:
        config.DISTILL = True

    ## set local rank for distributed training
    # will be set later
    #config.LOCAL_RANK = args.local_rank

    # output folder
    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.OUTPUT = os.path.join(config.OUTPUT, config.TAG)

    if not os.path.isdir(config.OUTPUT):
        os.makedirs(config.OUTPUT)

    # update DDP settings
    config.machine_rank = args.machine_rank
    config.num_nodes = args.num_machines
    config.dist_url = args.dist_url

    config.TRAIN.AUTO_RESUME = True

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
