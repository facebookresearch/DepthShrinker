# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from: https://github.com/rwightman/pytorch-image-models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import sys
from .helpers import to_2tuple

import time


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # N = (H//self.patch_size) * (H//self.patch_size)
        # x = x.reshape(B, C, H//self.patch_size, self.patch_size, H//self.patch_size, self.patch_size)
        # x = x.transpose(1, 3)
        # x = x.reshape(B, N, -1)

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, config=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)

#         self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

#         # N = (H//self.patch_size) * (H//self.patch_size)
#         # x = x.reshape(B, C, H//self.patch_size, self.patch_size, H//self.patch_size, self.patch_size)
#         # x = x.transpose(1, 3)
#         # x = x.reshape(B, N, -1)

#         x = self.proj(x)
#         x = self.pool(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x



def round2int(x):
    if type(x) is tuple:
        x = (int(x[0]), int(x[1]))
    else:
        x = int(x)

    return x


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temp=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temp, dim=-1)


def gumbel_softmax(logits, temp=1):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits, dim=-1), temp)

    return y


class Gumbel_Softmax:
    def __init__(self, temp=1):
        self.temp = temp

    def set_temp(self, temp):
        self.temp = temp

    def __call__(self, x):
        return gumbel_softmax(x, temp=self.temp)



class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)


    def forward(self, input, kernel_size, padding):
        weight = self.weight[:, :, :kernel_size[0], :kernel_size[1]]

        # print('weight shape:', weight.shape)

        output = F.conv2d(input, weight, self.bias, self.stride, padding, self.dilation, self.groups)

        # print('output shape:', output.shape)

        return output


class RPPatchEmbed_kernel_share(nn.Module):
    """ 2D Image to Regional Proposed Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, config=None):
        super().__init__()

        anchor_list = config.MODEL.VIT.ANCHOR_LIST
        act_layer = config.MODEL.VIT.PATCH_EMBED_ACT_LAYER
        rp_channel = config.MODEL.VIT.RP_CHANNEL
        sparse = config.MODEL.VIT.SPARSE
        ste_fun = config.MODEL.VIT.STE_FUN

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.sparse = sparse

        # large conv
        p = patch_size[0]

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        self.padding_list = [round2int((p*(anchor[0]-1)/2, p*(anchor[1]-1)/2)) for anchor in anchor_list]
        self.kernel_size_list = [round2int((p*anchor[0], p*anchor[1])) for anchor in anchor_list]

        max_kernel_size = round2int((p*max([anchor[0] for anchor in anchor_list]), p*max([anchor[1] for anchor in anchor_list])))

        self.proj_shared = USConv2d(in_chans, embed_dim, kernel_size=max_kernel_size, stride=patch_size)

        if act_layer == 'relu':
            self.act_layer = nn.ReLU()
        elif act_layer == 'lrelu':
            self.act_layer = nn.LeakyReLU()
        elif act_layer == 'hswish':
            self.act_layer = nn.Hardswish()
        else:
            print('No such activation func:', act_layer)
            sys.exit(0)

        if ste_fun == 'softmax':
            self.ste_fun = lambda x: F.softmax(x, dim=1)
        elif ste_fun == 'sigmoid':
            self.ste_fun = F.sigmoid
        elif ste_fun == 'gumbel_softmax':
            self.ste_fun = Gumbel_Softmax()
        else:
            print('No such STE func:', ste_fun)
            sys.exit(0)

        # pipeline RPN
        self.RPN1 = nn.Conv2d(in_chans, rp_channel[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(rp_channel[0])
        self.RPN2 = nn.Conv2d(rp_channel[0], rp_channel[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(rp_channel[1])
        self.RPN3 = nn.Conv2d(rp_channel[1], rp_channel[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(rp_channel[2])
        self.RPN4 = nn.Conv2d(rp_channel[2], self.anchor_num, kernel_size=3, stride=2, padding=1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def set_temp(self, temp):
        assert isinstance(self.ste_fun, Gumbel_Softmax)

        self.ste_fun.set_temp(temp)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."


        # get rpn classification score
        # rpn_conv1 = F.relu(self.RPN_Conv(origin_patch), inplace=True)
        # rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        embed_list = []
        for i in range(len(self.kernel_size_list)):
            # print('kernel_size:', self.kernel_size_list[i], 'padding size:', self.padding_list[i])
            # input()
            embed_list.append(self.proj_shared(x, kernel_size=self.kernel_size_list[i], padding=self.padding_list[i]))

        rpn_conv1 = self.act_layer(self.bn1(self.RPN1(x)))
        rpn_conv2 = self.act_layer(self.bn2(self.RPN2(rpn_conv1)))
        rpn_conv3 = self.act_layer(self.bn3(self.RPN3(rpn_conv2)))
        rpn_cls_score = self.RPN4(rpn_conv3)

        # rpn_cls_score_reshape = self.reshape(rpn_cls_score, self.anchor_num)

        if self.sparse:
            rpn_cls_score = rpn_cls_score.view(B, -1)  # (B, anchor_num * h * w)

            rpn_cls_one_hot = rpn_cls_score.clone()
            _, indices = rpn_cls_score.sort(dim=-1)

            j = int((1 - 1/self.anchor_num) * rpn_cls_score[0,:].numel())

            # flat_out = rpn_cls_one_hot.flatten(1)
            flat_out = rpn_cls_one_hot

            row_id1 = [[i for _ in range(j)] for i in range(B)]
            flat_out[row_id1, indices[:,:j]] = 0

            row_id2 = [[i for _ in range(indices.size(1)-j)] for i in range(B)]
            flat_out[row_id2, indices[:,j:]] = 1

            rpn_cls_score_softmax = self.ste_fun(rpn_cls_score)
            rpn_cls_score = (rpn_cls_one_hot - rpn_cls_score_softmax).detach() + rpn_cls_score_softmax

            embed_list_flat = torch.cat([torch.unsqueeze(embed, dim=2) for embed in embed_list], dim=2).view(B, self.embed_dim, -1).transpose(1,2)  # (B, anchor_num * h * w, embed_dim)

            merged_patch = embed_list_flat[row_id2, indices[:,j:], :] * rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1)   # (B, N, embed_dim)

            # print('j:', j)
            # print('indice:', indices[0,j:])
            # print('rpn_cls_one_hot:', rpn_cls_one_hot[0])
            # print('embed_list_flat.shape:', embed_list_flat.shape)
            # print('merged_patch.shape:', merged_patch.shape)
            # print('rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1):', rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1))
            # input()

        else:
            rpn_cls_prob = F.softmax(rpn_cls_score, 1)

            merged_patch = 0
            for i in range(self.anchor_num):
                merged_patch += torch.mul(embed_list[i].view(B, self.embed_dim, -1), rpn_cls_prob[:, i, :, :].view(B, 1, -1))

            merged_patch = merged_patch.transpose(1, 2) # BCN -> BNC

        merged_patch = self.norm(merged_patch)
        return merged_patch



class RPPatchEmbed(nn.Module):
    """ 2D Image to Regional Proposed Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, config=None):
        super().__init__()

        anchor_list = config.MODEL.VIT.ANCHOR_LIST
        act_layer = config.MODEL.VIT.PATCH_EMBED_ACT_LAYER
        rp_channel = config.MODEL.VIT.RP_CHANNEL
        sparse = config.MODEL.VIT.SPARSE
        ste_fun = config.MODEL.VIT.STE_FUN

        patch_embed_channel = config.MODEL.VIT.PATCH_EMBED_CHANNEL
        patch_embed_ks = config.MODEL.VIT.PATCH_EMBED_KS
        patch_embed_stride = config.MODEL.VIT.PATCH_EMBED_STRIDE

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.sparse = sparse

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        assert len(patch_embed_ks) == len(patch_embed_stride)
        assert len(patch_embed_channel) == len(patch_embed_stride) or len(patch_embed_channel) == len(patch_embed_stride) - 1

        proj_list = []
        for i in range(len(patch_embed_ks)):
            in_c = patch_embed_channel[i-1] if i>0 else in_chans
            out_c = patch_embed_channel[i] if i<len(patch_embed_channel)-1 else embed_dim
            ks = patch_embed_ks[i]
            stride = patch_embed_stride[i]
            padding = int(np.ceil((ks - stride) / 2))

            proj_list.append(nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding))

        self.proj = nn.Sequential(*proj_list)


        stride_total = 1
        for stride in patch_embed_stride:
            stride_total *= stride

        p = patch_size[0] // stride_total

        self.patch_num = img_size[0] // patch_size[0]

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        self.pooling_list = nn.ModuleList()
        for anchor in anchor_list:
            padding = round2int((p*(anchor[0]-1)/2, p*(anchor[1]-1)/2))
            kernel_size = round2int((p*anchor[0], p*anchor[1]))
            self.pooling_list.append(nn.AvgPool2d(kernel_size=kernel_size, stride=p, padding=padding))

        if act_layer == 'relu':
            self.act_layer = nn.ReLU()
        elif act_layer == 'lrelu':
            self.act_layer = nn.LeakyReLU()
        elif act_layer == 'hswish':
            self.act_layer = nn.Hardswish()
        else:
            print('No such activation func:', act_layer)
            sys.exit(0)

        if ste_fun == 'softmax':
            self.ste_fun = lambda x: F.softmax(x, dim=1)
        elif ste_fun == 'sigmoid':
            self.ste_fun = F.sigmoid
        elif ste_fun == 'gumbel_softmax':
            self.ste_fun = Gumbel_Softmax()
        else:
            print('No such STE func:', ste_fun)
            sys.exit(0)

        # pipeline RPN
        self.RPN1 = nn.Conv2d(in_chans, rp_channel[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(rp_channel[0])
        self.RPN2 = nn.Conv2d(rp_channel[0], rp_channel[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(rp_channel[1])
        self.RPN3 = nn.Conv2d(rp_channel[1], rp_channel[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(rp_channel[2])
        self.RPN4 = nn.Conv2d(rp_channel[2], self.anchor_num, kernel_size=3, stride=2, padding=1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def set_temp(self, temp):
        assert isinstance(self.ste_fun, Gumbel_Softmax)

        self.ste_fun.set_temp(temp)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # t1 = time.time()

        embedding = self.proj(x)

        # print("embedding.shape", embedding.shape)

        embed_list = []
        for pool_layer in self.pooling_list:
            # print('anchor shape:', pool_layer(embedding).shape)
            embed_list.append(pool_layer(embedding))

        rpn_conv1 = self.act_layer(self.bn1(self.RPN1(x)))
        rpn_conv2 = self.act_layer(self.bn2(self.RPN2(rpn_conv1)))
        rpn_conv3 = self.act_layer(self.bn3(self.RPN3(rpn_conv2)))
        rpn_cls_score = self.RPN4(rpn_conv3)

        # t2 = time.time()
        # print('rpn time:', t2-t1)

        if self.sparse:
            rpn_cls_score = rpn_cls_score.view(B, -1)  # (B, anchor_num * h * w)

            rpn_cls_one_hot = rpn_cls_score.clone()
            _, indices = rpn_cls_score.sort(dim=-1)

            j = int((1 - 1/self.anchor_num) * rpn_cls_score[0,:].numel())

            # flat_out = rpn_cls_one_hot.flatten(1)
            flat_out = rpn_cls_one_hot

            row_id1 = [[i for _ in range(j)] for i in range(B)]
            flat_out[row_id1, indices[:,:j]] = 0

            row_id2 = [[i for _ in range(indices.size(1)-j)] for i in range(B)]
            flat_out[row_id2, indices[:,j:]] = 1

            rpn_cls_score_softmax = self.ste_fun(rpn_cls_score)
            rpn_cls_score = (rpn_cls_one_hot - rpn_cls_score_softmax).detach() + rpn_cls_score_softmax

            embed_list_flat = torch.cat([torch.unsqueeze(embed, dim=2) for embed in embed_list], dim=2).view(B, self.embed_dim, -1).transpose(1,2)  # (B, anchor_num * h * w, embed_dim)

            merged_patch = embed_list_flat[row_id2, indices[:,j:], :] * rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1)   # (B, N, embed_dim)


        else:
            rpn_cls_prob = F.softmax(rpn_cls_score, 1)

            merged_patch = 0
            for i in range(self.anchor_num):
                merged_patch += torch.mul(embed_list[i].view(B, self.embed_dim, -1), rpn_cls_prob[:, i, :, :].view(B, 1, -1))

            merged_patch = merged_patch.transpose(1, 2) # BCN -> BNC

        merged_patch = self.norm(merged_patch)
        return merged_patch



class RPPatchEmbed_reshape(nn.Module):
    """ 2D Image to Regional Proposed Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, config=None):
        super().__init__()

        anchor_list = config.MODEL.VIT.ANCHOR_LIST
        act_layer = config.MODEL.VIT.PATCH_EMBED_ACT_LAYER
        rp_channel = config.MODEL.VIT.RP_CHANNEL
        sparse = config.MODEL.VIT.SPARSE
        ste_fun = config.MODEL.VIT.STE_FUN

        patch_embed_channel = config.MODEL.VIT.PATCH_EMBED_CHANNEL
        patch_embed_ks = config.MODEL.VIT.PATCH_EMBED_KS
        patch_embed_stride = config.MODEL.VIT.PATCH_EMBED_STRIDE

        pooling_size = config.MODEL.VIT.POOLING_SIZE

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.sparse = sparse

        self.anchor_list = anchor_list

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        assert len(patch_embed_ks) == len(patch_embed_stride)
        assert len(patch_embed_channel) == len(patch_embed_stride) or len(patch_embed_channel) == len(patch_embed_stride) - 1

        proj_list = []
        for i in range(len(patch_embed_ks)):
            in_c = patch_embed_channel[i-1] if i>0 else in_chans
            out_c = patch_embed_channel[i] if i<len(patch_embed_channel)-1 else self.embed_dim // pooling_size // pooling_size
            ks = patch_embed_ks[i]
            stride = patch_embed_stride[i]
            padding = int(np.ceil((ks - stride) / 2))

            proj_list.append(nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding))

        self.proj = nn.Sequential(*proj_list)


        stride_total = 1
        for stride in patch_embed_stride:
            stride_total *= stride

        p = patch_size[0] // stride_total

        self.patch_num = img_size[0] // patch_size[0]

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        assert patch_embed_channel[-1] * pooling_size * pooling_size == self.embed_dim
        self.adaptive_pooling = nn.AdaptiveMaxPool2d(pooling_size)
        self.pooling_size = pooling_size

        self.unfold_list = nn.ModuleList()
        for anchor in anchor_list:
            padding = round2int((p*(anchor[0]-1)/2, p*(anchor[1]-1)/2))
            kernel_size = round2int((p*anchor[0], p*anchor[1]))

            self.unfold_list.append(nn.Unfold(kernel_size=kernel_size, padding=padding, stride=p))

        self.p = p

        if act_layer == 'relu':
            self.act_layer = nn.ReLU()
        elif act_layer == 'lrelu':
            self.act_layer = nn.LeakyReLU()
        elif act_layer == 'hswish':
            self.act_layer = nn.Hardswish()
        else:
            print('No such activation func:', act_layer)
            sys.exit(0)

        if ste_fun == 'softmax':
            self.ste_fun = lambda x: F.softmax(x, dim=1)
        elif ste_fun == 'sigmoid':
            self.ste_fun = F.sigmoid
        elif ste_fun == 'gumbel_softmax':
            self.ste_fun = Gumbel_Softmax()
        else:
            print('No such STE func:', ste_fun)
            sys.exit(0)

        # pipeline RPN
        self.RPN1 = nn.Conv2d(in_chans, rp_channel[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(rp_channel[0])
        self.RPN2 = nn.Conv2d(rp_channel[0], rp_channel[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(rp_channel[1])
        self.RPN3 = nn.Conv2d(rp_channel[1], rp_channel[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(rp_channel[2])
        self.RPN4 = nn.Conv2d(rp_channel[2], self.anchor_num, kernel_size=3, stride=2, padding=1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def set_temp(self, temp):
        assert isinstance(self.ste_fun, Gumbel_Softmax)

        self.ste_fun.set_temp(temp)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # t1 = time.time()

        embedding = self.proj(x)

        B, C, H, W = embedding.shape

        # print("embedding.shape", embedding.shape)

        embed_list = []
        for i, unfold_op in enumerate(self.unfold_list):
            embed = unfold_op(embedding)  # (B, C * kernel_size * kernel_size, patch_num * patch_num), need to downsample C *kernel_size * kernel_size to C * p * p
            # print('After unfold:', embed.shape)
            embed = embed.transpose(1, 2).reshape(B, self.patch_num * self.patch_num, C, self.p * self.anchor_list[i][0], self.p * self.anchor_list[i][1])  # (B, N, C, kernel_size, kernel_size)
            # print('After reshape, before pooling:', embed.shape)
            embed = self.adaptive_pooling(embed.reshape(B, self.patch_num * self.patch_num * C, self.p * self.anchor_list[i][0], self.p * self.anchor_list[i][1])).reshape(B, self.patch_num * self.patch_num, C * self.pooling_size * self.pooling_size)  # (B, N, C)
            # print('After pooling:', embed.shape)
            # input()
            embed_list.append(embed)


        rpn_conv1 = self.act_layer(self.bn1(self.RPN1(x)))
        rpn_conv2 = self.act_layer(self.bn2(self.RPN2(rpn_conv1)))
        rpn_conv3 = self.act_layer(self.bn3(self.RPN3(rpn_conv2)))
        rpn_cls_score = self.RPN4(rpn_conv3)

        # t2 = time.time()
        # print('rpn time:', t2-t1)

        if self.sparse:
            rpn_cls_score = rpn_cls_score.view(B, -1)  # (B, anchor_num * h * w)

            rpn_cls_one_hot = rpn_cls_score.clone()
            _, indices = rpn_cls_score.sort(dim=-1)

            j = int((1 - 1/self.anchor_num) * rpn_cls_score[0,:].numel())

            # flat_out = rpn_cls_one_hot.flatten(1)
            flat_out = rpn_cls_one_hot

            row_id1 = [[i for _ in range(j)] for i in range(B)]
            flat_out[row_id1, indices[:,:j]] = 0

            row_id2 = [[i for _ in range(indices.size(1)-j)] for i in range(B)]
            flat_out[row_id2, indices[:,j:]] = 1

            rpn_cls_score_softmax = self.ste_fun(rpn_cls_score)
            rpn_cls_score = (rpn_cls_one_hot - rpn_cls_score_softmax).detach() + rpn_cls_score_softmax

            embed_list_flat = torch.cat([torch.unsqueeze(embed, dim=2) for embed in embed_list], dim=2).view(B, -1, self.embed_dim)  # (B, anchor_num * h * w, embed_dim)

            merged_patch = embed_list_flat[row_id2, indices[:,j:], :] * rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1)   # (B, N, embed_dim)


        else:
            rpn_cls_prob = F.softmax(rpn_cls_score, 1)

            merged_patch = 0
            for i in range(self.anchor_num):
                merged_patch += torch.mul(embed_list[i], rpn_cls_prob[:, i, :, :].view(B, -1, 1))

        merged_patch = self.norm(merged_patch)
        return merged_patch




class RPPatchEmbed_roi_align(nn.Module):
    """ 2D Image to Regional Proposed Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, config=None):
        super().__init__()

        anchor_list = config.MODEL.VIT.ANCHOR_LIST
        act_layer = config.MODEL.VIT.PATCH_EMBED_ACT_LAYER
        rp_channel = config.MODEL.VIT.RP_CHANNEL
        sparse = config.MODEL.VIT.SPARSE
        ste_fun = config.MODEL.VIT.STE_FUN

        patch_embed_channel = config.MODEL.VIT.PATCH_EMBED_CHANNEL
        patch_embed_ks = config.MODEL.VIT.PATCH_EMBED_KS
        patch_embed_stride = config.MODEL.VIT.PATCH_EMBED_STRIDE

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.sparse = sparse

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        assert len(patch_embed_ks) == len(patch_embed_stride)
        assert len(patch_embed_channel) == len(patch_embed_stride) or len(patch_embed_channel) == len(patch_embed_stride) - 1

        proj_list = []
        for i in range(len(patch_embed_ks)):
            in_c = patch_embed_channel[i-1] if i>0 else in_chans
            out_c = patch_embed_channel[i] if i<len(patch_embed_channel)-1 else embed_dim
            ks = patch_embed_ks[i]
            stride = patch_embed_stride[i]
            padding = int(np.ceil((ks - stride) / 2))

            proj_list.append(nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding))

        self.proj = nn.Sequential(*proj_list)


        stride_total = 1
        for stride in patch_embed_stride:
            stride_total *= stride

        p = patch_size[0] // stride_total

        patch_num = img_size[0] // patch_size[0]

        roi_box_all = []
        for anchor in anchor_list:
            h = anchor[0] * p
            w = anchor[1] * p

            roi_box_list = []

            for i in range(patch_num):
                for j in range(patch_num):
                    centor = [(p-1)/2 + i*p,(p-1)/2 + j*p]
                    roi_box = [centor[0]-h/2, centor[1]-w/2, centor[0]+h/2, centor[1]+w/2]

                    roi_box_list.append(roi_box)

            roi_box_all.extend(roi_box_list)
            # roi_box_all.append(roi_box_list)

        self.roi_box_all = roi_box_all  # [anchor_num * patch_num * patch_num, 4]
        # self.roi_box_all = torch.tensor(roi_box_all) # [anchor_num, patch_num * patch_num, 4]
        # self.roi_box_all = self.roi_box_all.view(-1, 4) # [anchor_num * patch_num * patch_num, 4]

        if act_layer == 'relu':
            self.act_layer = nn.ReLU()
        elif act_layer == 'lrelu':
            self.act_layer = nn.LeakyReLU()
        elif act_layer == 'hswish':
            self.act_layer = nn.Hardswish()
        else:
            print('No such activation func:', act_layer)
            sys.exit(0)

        if ste_fun == 'softmax':
            self.ste_fun = lambda x: F.softmax(x, dim=1)
        elif ste_fun == 'sigmoid':
            self.ste_fun = F.sigmoid
        else:
            print('No such STE func:', ste_fun)
            sys.exit(0)

        # pipeline RPN
        self.RPN1 = nn.Conv2d(in_chans, rp_channel[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(rp_channel[0])
        self.RPN2 = nn.Conv2d(rp_channel[0], rp_channel[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(rp_channel[1])
        self.RPN3 = nn.Conv2d(rp_channel[1], rp_channel[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(rp_channel[2])
        self.RPN4 = nn.Conv2d(rp_channel[2], self.anchor_num, kernel_size=3, stride=2, padding=1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def set_temp(self, temp):
        assert isinstance(self.ste_fun, Gumbel_Softmax)

        self.ste_fun.set_temp(temp)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # t1 = time.time()

        embedding = self.proj(x)

        rpn_conv1 = self.act_layer(self.bn1(self.RPN1(x)))
        rpn_conv2 = self.act_layer(self.bn2(self.RPN2(rpn_conv1)))
        rpn_conv3 = self.act_layer(self.bn3(self.RPN3(rpn_conv2)))
        rpn_cls_score = self.RPN4(rpn_conv3)

        # t2 = time.time()
        # print('rpn time:', t2-t1)

        if self.sparse:
            rpn_cls_score = rpn_cls_score.view(B, -1)  # (B, anchor_num * h * w)

            rpn_cls_one_hot = rpn_cls_score.clone()
            _, indices = rpn_cls_score.sort(dim=-1)  # [B, anchor_num * h * w]

            j = int((1 - 1/self.anchor_num) * rpn_cls_score[0,:].numel())

            # flat_out = rpn_cls_one_hot.flatten(1)
            flat_out = rpn_cls_one_hot

            row_id1 = [[i for _ in range(j)] for i in range(B)]
            flat_out[row_id1, indices[:,:j]] = 0

            row_id2 = [[i for _ in range(indices.size(1)-j)] for i in range(B)]
            flat_out[row_id2, indices[:,j:]] = 1

            rpn_cls_score_softmax = self.ste_fun(rpn_cls_score)
            rpn_cls_score = (rpn_cls_one_hot - rpn_cls_score_softmax).detach() + rpn_cls_score_softmax

            roi_box_all = torch.tensor(self.roi_box_all, device=x.device)
            roi_box_list = [roi_box_all[indices[i,j:]] for i in range(B)]

            embedding = torchvision.ops.roi_align(embedding, boxes=roi_box_list, output_size=[1,1])  # (B*N, C, 1, 1)

            embedding = embedding.squeeze().view(B, -1, self.embed_dim)  # (B, N, C)

            # print('embedding.shape', embedding.shape)

            ### test the correctness of roi_align
            # roi_box_all = torch.tensor(self.roi_box_all, device=x.device)
            # roi_box_list = [roi_box_all[:196] for _ in range(B)]
            # embedding = torchvision.ops.roi_align(x, boxes=roi_box_list, output_size=[16,16])  # (B*N, 3, 16, 16)
            # embedding = embedding.view(B, 14, 14, 3, 16, 16).permute(0, 3, 2, 4, 1, 5).contiguous().view(B, 3, 224, 224)

            # for i in range(B):
            #     torchvision.utils.save_image(embedding[i], 'on_device_ai/Tools/experimental/swin_transformer/img/%d.jpg'%i)
            #     torchvision.utils.save_image(x[i], 'on_device_ai/Tools/experimental/swin_transformer/img/%d_orig.jpg'%i)
            # input()

            output = embedding * rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1)

        else:
            rpn_cls_prob = F.softmax(rpn_cls_score, 1) # (B, anchor_num, h, w)
            rpn_cls_prob = rpn_cls_prob.view(B, self.anchor_num, -1)  # (B, anchor_num, N)

            roi_box_all = torch.tensor(self.roi_box_all, device=x.device)
            embedding = torchvision.ops.roi_align(embedding, boxes=[roi_box_all for _ in range(B)], output_size=[1,1])  # (B * anchor_num * patch_num * patch_num, C, 1, 1)
            embedding = embedding.view(B, self.anchor_num, -1, self.embed_dim)  # (B, anchor_num, N, C)

            # print('embedding.shape', embedding.shape)

            output = 0
            for i in range(self.anchor_num):
                output += embedding[:, i, :, :] * rpn_cls_prob[:, i, :].view(B, -1, 1)


        # t3 = time.time()
        # print('roi align time:', t2-t1)

        output = self.norm(output)

        return output





class RPPatchEmbed_no_share(nn.Module):
    """ 2D Image to Regional Proposed Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, config=None):
        super().__init__()

        anchor_list = config.MODEL.VIT.ANCHOR_LIST
        act_layer = config.MODEL.VIT.PATCH_EMBED_ACT_LAYER
        rp_channel = config.MODEL.VIT.RP_CHANNEL
        sparse = config.MODEL.VIT.SPARSE
        ste_fun = config.MODEL.VIT.STE_FUN

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.sparse = sparse

        # large conv
        p = patch_size[0]

        assert anchor_list is not None
        self.anchor_num = len(anchor_list)

        self.proj_list = nn.ModuleList()
        for anchor in anchor_list:
            padding = round2int((p*(anchor[0]-1)/2, p*(anchor[1]-1)/2))
            self.proj_list.append(nn.Conv2d(in_chans, embed_dim, kernel_size=round2int((p*anchor[0], p*anchor[1])), stride=patch_size, padding=padding))


        if act_layer == 'relu':
            self.act_layer = nn.ReLU()
        elif act_layer == 'lrelu':
            self.act_layer = nn.LeakyReLU()
        elif act_layer == 'hswish':
            self.act_layer = nn.Hardswish()
        else:
            print('No such activation func:', act_layer)
            sys.exit(0)

        if ste_fun == 'softmax':
            self.ste_fun = lambda x: F.softmax(x, dim=1)
        elif ste_fun == 'sigmoid':
            self.ste_fun = F.sigmoid
        elif ste_fun == 'gumbel_softmax':
            self.ste_fun = Gumbel_Softmax()
        else:
            print('No such STE func:', ste_fun)
            sys.exit(0)

        # pipeline RPN
        self.RPN1 = nn.Conv2d(in_chans, rp_channel[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(rp_channel[0])
        self.RPN2 = nn.Conv2d(rp_channel[0], rp_channel[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(rp_channel[1])
        self.RPN3 = nn.Conv2d(rp_channel[1], rp_channel[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(rp_channel[2])
        self.RPN4 = nn.Conv2d(rp_channel[2], self.anchor_num, kernel_size=3, stride=2, padding=1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def set_temp(self, temp):
        assert isinstance(self.ste_fun, Gumbel_Softmax)

        self.ste_fun.set_temp(temp)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."


        # get rpn classification score
        # rpn_conv1 = F.relu(self.RPN_Conv(origin_patch), inplace=True)
        # rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        embed_list = []
        for proj in self.proj_list:
            embed_list.append(proj(x))

        rpn_conv1 = self.act_layer(self.bn1(self.RPN1(x)))
        rpn_conv2 = self.act_layer(self.bn2(self.RPN2(rpn_conv1)))
        rpn_conv3 = self.act_layer(self.bn3(self.RPN3(rpn_conv2)))
        rpn_cls_score = self.RPN4(rpn_conv3)

        # rpn_cls_score_reshape = self.reshape(rpn_cls_score, self.anchor_num)

        if self.sparse:
            rpn_cls_score = rpn_cls_score.view(B, -1)  # (B, anchor_num * h * w)

            rpn_cls_one_hot = rpn_cls_score.clone()
            _, indices = rpn_cls_score.sort(dim=-1)

            j = int((1 - 1/self.anchor_num) * rpn_cls_score[0,:].numel())

            # flat_out = rpn_cls_one_hot.flatten(1)
            flat_out = rpn_cls_one_hot

            row_id1 = [[i for _ in range(j)] for i in range(B)]
            flat_out[row_id1, indices[:,:j]] = 0

            row_id2 = [[i for _ in range(indices.size(1)-j)] for i in range(B)]
            flat_out[row_id2, indices[:,j:]] = 1

            rpn_cls_score_softmax = self.ste_fun(rpn_cls_score)
            rpn_cls_score = (rpn_cls_one_hot - rpn_cls_score_softmax).detach() + rpn_cls_score_softmax

            embed_list_flat = torch.cat([torch.unsqueeze(embed, dim=2) for embed in embed_list], dim=2).view(B, self.embed_dim, -1).transpose(1,2)  # (B, anchor_num * h * w, embed_dim)

            merged_patch = embed_list_flat[row_id2, indices[:,j:], :] * rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1)   # (B, N, embed_dim)

            # print('j:', j)
            # print('indice:', indices[0,j:])
            # print('rpn_cls_one_hot:', rpn_cls_one_hot[0])
            # print('embed_list_flat.shape:', embed_list_flat.shape)
            # print('merged_patch.shape:', merged_patch.shape)
            # print('rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1):', rpn_cls_score[row_id2, indices[:,j:]].view(B, -1, 1))
            # input()

        else:
            rpn_cls_prob = F.softmax(rpn_cls_score, 1)

            merged_patch = 0
            for i in range(self.anchor_num):
                merged_patch += torch.mul(embed_list[i].view(B, self.embed_dim, -1), rpn_cls_prob[:, i, :, :].view(B, 1, -1))

            merged_patch = merged_patch.transpose(1, 2) # BCN -> BNC

        merged_patch = self.norm(merged_patch)
        return merged_patch
