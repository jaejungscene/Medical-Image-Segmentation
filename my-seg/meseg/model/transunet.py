from os.path import join as pjoin
from collections import OrderedDict

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch.nn.modules.utils import _pair
from scipy import ndimage
# from .inception_transformer import InceptionBlock
from .weight_init import trunc_normal_
# print("asdfsdf")
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)



class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))



class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1] # feature inverse



class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        num_heads = config.hidden_size//64
        num_heads = 1 if num_heads==0 else num_heads
        self.num_attention_heads = num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size*4)
        self.fc2 = nn.Linear(config.hidden_size*4, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)

            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        trunc_normal_(self.position_embeddings, std=.02)
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x) # (B, 1024, 14, 14)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))

        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, features



class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class InceptionBlock(nn.Module):
    r"""
    num_path: a factor that determines how many inception paths are created.
        so, num_path 1 means gerneral attention.
    direction: if direction is 0, path is odd and there are both pixshuf & unpixshuf
               if direction is 1, there are only unpixshuf path (channel up, spatial size down)
               if direction is 2, there are only pixshuf path (channel down, spatial size up)
    """
    def __init__(self, config, dim:int=1, num_heads:int=1, mlp_ratio:float=4.,
                 qkv_bias:bool=True, num_path:int=1, pixshuf_factor:int=1,
                 drop_rate:float=0., attn_drop:float=0., proj_drop:float=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, concat:bool=True, direction:int=0) -> None:
        super(InceptionBlock, self).__init__()
        # assert num_path%2==1, f"num_path must be odd number"
        assert direction==0 or direction==1 or direction==2, "direction must be 0 or 1 or 2."
        if (direction==0 and num_path%2==0):
            assert False, "if direction is 0, num_path must be odd number."
        
        self.dim = config.hidden_size
        self.num_path = config.num_path
        self.median = (self.num_path//2)+1 if self.num_path%2==1 else self.num_path//2
        self.total_dim = self.dim * self.num_path
        self.pixshuf_factor = config.pixshuf_factor
        self.direction = config.direction
        self.concat = config.concat
        # self.pos_embed = config.pos_embed
        self.make_path(config)
        if self.concat:
            self.norm = norm_layer(self.total_dim)
            self.fc = nn.Linear(self.total_dim, self.dim)
    

    def make_path(self, config)->None:
        for i in range(1, self.num_path+1):
            # print(self.median)
            if self.direction == 0:
                if i < self.median:
                    new_dim = self.dim * (self.pixshuf_factor**(i*2))
                elif i > self.median:   
                    new_dim = int(self.dim / (self.pixshuf_factor**((i-self.median)*2)))
                else:   new_dim = self.dim
            elif self.direction == 1:
                if i==1:    new_dim = self.dim
                else:   new_dim = self.dim * (self.pixshuf_factor**((i-1)*2))
            else:
                if i==1:    new_dim = self.dim
                else:   new_dim = int(self.dim / (self.pixshuf_factor**((i-1)*2)))
            # print(new_dim)
            config.hidden_size = new_dim
            # print(">>>>>>>> ",config.hidden_size)
            self.add_module(f"path{i}", Block(config, False))
            # self.total_dim = self.total_dim + new_dim
        # raise Exception("----------end-----------")
        config.hidden_size = self.dim

    def _forward_each_paths(self, x:torch.Tensor) -> torch.Tensor:
        features = []
        flag = True
        for i in range(1, self.num_path+1):
            if self.direction == 0:
                if i < self.median:
                    shuf1 = nn.PixelUnshuffle(self.pixshuf_factor*i)
                    shuf2 = nn.PixelShuffle(self.pixshuf_factor*i)
                elif i > self.median:
                    shuf1 = nn.PixelShuffle(self.pixshuf_factor*(i-self.median))
                    shuf2 = nn.PixelUnshuffle(self.pixshuf_factor*(i-self.median))
                else:
                    flag = False
            elif self.direction == 1: # only up channel
                if i==1:
                    flag = False
                else:   
                    shuf1 = nn.PixelUnshuffle(self.pixshuf_factor*(i-1))
                    shuf2 = nn.PixelShuffle(self.pixshuf_factor*(i-1))
            else: # only down channel 
                if i==1:
                    flag = False
                else:   
                    shuf1 = nn.PixelShuffle(self.pixshuf_factor*(i-1))
                    shuf2 = nn.PixelUnshuffle(self.pixshuf_factor*(i-1))

            x = shuf1(x) if flag else x
            B, D, H, W = x.shape
            x = x.contiguous().view(B,D,H*W).permute(0,2,1)
            x, attn_probs = getattr(self, f"path{i}")(x)
            x = x.permute(0,2,1).view(B,D,H,W)
            x = shuf2(x) if flag else x

            features.append(x)
            flag = True
        return features, attn_probs

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = W = int(np.sqrt(N))
        x = x.permute(0,2,1).view(B,D,H,W)
        features, attn_probs = self._forward_each_paths(x)
        if self.concat:
            x = torch.cat(features,dim=1)
            x = x.view(B,self.total_dim,N).permute(0,2,1)
            x = self.fc(self.norm(x))
        else:
            x = features[0]
            for f in features[1:]:
                x = x + f
            x = x.view(B,self.dim,N).permute(0,2,1)
        
        return x, attn_probs



class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            if config.block == "normal":
                layer = Block(config, vis)
            elif config.block == "inception":   # <---------------------- changed block in transunet to inception transformer block
                layer = InceptionBlock(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights



class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # print("=================")
        # print("embeddings shape: ", embedding_output.shape)
        # print("features: ", len(features))
        # print("=================")
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        # print("===================")
        # print("x shape: ", x.shape)
        # if skip is not None:
        #     print("skip: ", skip.shape)
        # print("===================")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        # print("==========================")
        # print("in channel: ", in_channels)
        # print("kernel size: ", kernel_size)
        # print("out channel: ", out_channels)
        # print("==========================")
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)



class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w) # x = x.reshape(B, hidden, h, w)
        
        # print("=======================")
        # print("input shape in decoder: ", x.shape)
        # print("=======================")

        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x



class TransUnet(nn.Module):
    def __init__(self, config, vis=False):
        super(TransUnet, self).__init__()
        self.classifier = config.classifier
        self.transformer = Transformer(config, config.img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  
        # print("==========")
        # print("x shape: ", x.shape) # (B, n_patch, hidden)
        # print("==========")
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            # if self.config.block == "normal":
            if False: 
            # when ImageNet pre-train weights is not loaded to transformer, TransUnet performs better.
                print("loading vit pretrained weights.")
                self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
                self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

                posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

                posemb_new = self.transformer.embeddings.position_embeddings
                if posemb.size() == posemb_new.size():
                    self.transformer.embeddings.position_embeddings.copy_(posemb)
                elif posemb.size()[1]-1 == posemb_new.size()[1]:
                    posemb = posemb[:, 1:]
                    self.transformer.embeddings.position_embeddings.copy_(posemb)
                else:
                    logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                    ntok_new = posemb_new.size(1)
                    if self.classifier == "seg":
                        _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    gs_old = int(np.sqrt(len(posemb_grid)))
                    gs_new = int(np.sqrt(ntok_new))
                    print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb = posemb_grid
                    self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

                # Encoder whole
                for bname, block in self.transformer.encoder.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)
            print("loading resnet pretrained weights.")
            if self.transformer.embeddings.hybrid:
                res_weight = weights
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                

# if __name__ == "__main__":
#     class args:
#         ## type of model
#         model_type = None

#         ## name of model
#         # model_name = "transunet_pos"                                    #105,277,081(class=9)

#         # model_name = "inception-transunet_d192_p1_d0_f2_add"        #22,202,329

#         # model_name = "inception-transunet_d192_p3_d0_f2_add"        #107,596,057
#         # model_name = "inception-transunet_d192_p3_d0_f2_concat"     #108,939,289
#         # model_name = "inception-transunet_d128_p3_d0_f2_concat"     #108,939,289
#         # model_name = "inception-transunet_d128_p3_d0_f2_add"     #108,939,289
#         # model_name = "inception-transunet_d128_p3_d0_f2_add_pos"        #107,596,057 -- 14*12=168

#         # model_name = "inception-transunet_d192_p2_d1_f2_add"        #107,256,793
#         # model_name = "inception-transunet_d192_p2_d1_f2_concat"     #108,153,049
#         # model_name = "inception-transunet_d128_p2_d1_f2_concat"     #108,153,049
#         # model_name = "inception-transunet_d128_p2_d1_f2_add"     #108,153,049

#         # model_name = "inception-transunet_d192_p2_d2_f2_add"        #22,541,593
#         # model_name = "inception-transunet_d192_p2_d2_f2_concat"     #23,437,849
#         model_name = "inception-transunet_d168_p2_d2_f2_concat"     #23,437,849
#         model_name = "inception-transunet_d168_p2_d2_f2_add"        #22,541,593

#         mode = 'test'
#         img_size = 224
#         num_classes = 9
#         device = "cuda:4"
#         pretrained = False

#     inputs = torch.randn((2,1,224,224)).cuda(args.device)

#     import model_config
#     config = model_config.transunet()
#         # print(config)
#         # assert False, "-------- end ---------"
#     config.n_classes = args.num_classes
#     config.img_size = args.img_size
#     modelconfig = args.model_name.split("_")
#     if args.model_name.startswith("transunet"):
#         config.block = "normal"
#         model = TransUnet(config).cuda(args.device)
#     else: # inception-transunet_d{hidden_size}_p{num_path}_d{direction}_f{pixshuf_factor}_{concat}
#         config.block = "inception"
#         config.hidden_size = int(modelconfig[1][1:]) # 768, 384, 192
#         # config.transformer["mlp_dim"] = config.hidden_size*4
#         # config.transformer["num_heads"] = config.hidden_size//64
#         config.num_path = int(modelconfig[2][1:])
#         config.direction = int(modelconfig[3][1:])
#         config.pixshuf_factor = int(modelconfig[4][1:])
#         config.concat = True if modelconfig[5]=="concat" else False
#         # print(config.hidden_size)
#         # print(config.num_path)
#         # print(config.direction)
#         # print(config.pixshuf_factor)
#         # print(config.concat)
#         # assert False, "====== end ======="
#         config.pos_embed = False
#         model = TransUnet(config).cuda(args.device)

#     # model.load_state_dict(torch.load("/home/ljj0512/private/workspace/Medical/segmentation/log/btcv_2d_inception-transunet_d192_p3_d0_f2_concat_pos_v1/best_weight.pth"))
#     print('#params of TransUnet: {:,}'.format(sum([p.data.nelement() for p in model.parameters()])))
#     print("outputs shape: ",model(inputs).shape)