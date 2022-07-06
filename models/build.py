# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# --------------------------------------------------------
# Modified from Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import os

from .factory import create_model

# for requests
os.environ['HTTP_PROXY'] = "http://fwdproxy:8080"
os.environ['HTTPS_PROXY'] = "https://fwdproxy:8080"
# for urllib
os.environ['http_proxy'] = "fwdproxy:8080"
os.environ['https_proxy'] = "fwdproxy:8080"


def build_model(config):
    model_type = config.MODEL.TYPE

    if 'resnet' in model_type:
        model = create_model(
            model_type,
            # pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            drop_block_rate=config.MODEL.DROP_BLOCK_RATE,
            config=config,
        )
    elif 'vgg' in model_type:
        model = create_model(
            model_type,
            # pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            config=config,
        )
    elif 'efficientnet' in model_type or 'mobilenetv2' in model_type:
        model = create_model(
            model_type,
            # pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            config=config,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
