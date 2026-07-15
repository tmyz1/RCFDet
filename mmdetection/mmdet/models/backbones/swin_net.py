import math
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.backbones.swin import *
import sys
from mmcv.cnn.bricks.transformer import FFN
from typing import List
from ..layers import PatchMerging
from torch import Tensor
from mmdet.models.necks import YOLOXPAFPN


@MODELS.register_module()
class SwinNet(BaseModule):
    def __init__(self,in_channels:int,embed_dims:int,depths:tuple[int],num_heads:tuple[int],window_size:int,
                 mlp_ratio:int,qkv_bias,drop_path_rate:float,out_indices:tuple[int],
                 pretrained=None,init_cfg=None,act_cfg=dict(type='GELU')):
        #Swin 部分
        super(SwinNet, self).__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.pretrained = pretrained

        self.CRFuse=CrossResidualFusion(
            in_channels = self.in_channels,
            embed_dims=self.embed_dims,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_path_rate=self.drop_path_rate,
            out_indices=self.out_indices,
            pretrained=self.pretrained,
            init_cfg=self.init_cfg,
            act_cfg=act_cfg,
            with_img_mask=True
        )

    def forward(self,x):
        rgb,ir,a=x
        self.image_vis(rgb,'rgb')
        sys.exit(0)
        rgb_fuses,ir_fuses=self.CRFuse(rgb, ir, a)
        return rgb_fuses,ir_fuses,a

    def image_vis(self,image,name):
        image = image.permute(0,2,3,1)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        temp = image[0].detach().cpu().numpy()
        temp = np.uint8((temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-8) * 255)
        cv2.imshow(name, temp)
        cv2.waitKey(0)

class CrossResidualFusion(nn.Module):
    def __init__(self,in_channels:int,embed_dims:int,depths:tuple[int],num_heads:tuple[int],window_size:int,
                 mlp_ratio:int,qkv_bias,drop_path_rate:float,out_indices:tuple[int],
                 pretrained=None,init_cfg=None,act_cfg=dict(type='GELU'),with_img_mask=True):
        super().__init__()
        #构造Swin部分
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.pretrained = pretrained
        self.rgb_swin, self.ir_swin = self.build_SwinTransformer()
        self.with_img_mask = with_img_mask
        #load_swin_to_submodule(self.rgb_swin, r'/root/autodl-tmp/weight/swin_tiny_224.pth')
        load_swin_to_submodule(self.rgb_swin, r"D:\pycharm work\weight\swin_tiny_224.pth")
        self.ir_swin.load_state_dict(self.rgb_swin.state_dict(), strict=False)  # 复制给另一个分支

        self.downsample = ModuleList()#下采样模块
        self.qkv = ModuleList()#每一层的qkv的非线性层
        self.proj = ModuleList()
        self.proj_drop = nn.Dropout(0.1)
        #模态之间的注意力计算
        self.dual_attn=Dual_Mode_Attention(embed_dims=self.embed_dims,depths=self.depths,dropout=0.1)
        for i in range(0, len(self.depths)):
            self.downsample.append(
                PatchMerging(
                    in_channels=self.embed_dims * (2 ** i),
                    out_channels=self.embed_dims * (2 ** i) * 2,
                    stride=2,
                    norm_cfg=dict(type='LN'),
                    init_cfg=None)
            )
            self.qkv.append(
                nn.Linear(self.embed_dims * (2 ** i), self.embed_dims * (2 ** i)*3,bias=True)
            )
            self.proj.append(
                nn.Linear(self.embed_dims * (2 ** i), self.embed_dims * (2 ** i))
            )

    def forward(self, rgb, ir, a):
        rgb_hw_shape = ir_hw_shape = None
        rgb_feats: List[Tensor] = []
        ir_feats: List[Tensor] = []

        for i in range(0, len(self.depths)):
            rgb, rgb_hw_shape, rgb_out, _, _ = self.rgb_swin.forward_stage(rgb, i, rgb_hw_shape)
            ir, ir_hw_shape, ir_out, _, _ = self.ir_swin.forward_stage(ir, i, ir_hw_shape)
            rgb_feats.append(rgb_out)
            ir_feats.append(ir_out)

        a = a.to(device=rgb_feats[0].device, dtype=rgb_feats[0].dtype).view(-1)
        dual_mask = (a > 0) & (a < 1)

        if dual_mask.any():
            rgb_fuses = self.dual_attn(rgb_feats, ir_feats)
            rgb_only_mask = (a >= 1).view(-1, 1, 1, 1)
            if rgb_only_mask.any():
                rgb_fuses = [
                    torch.where(rgb_only_mask, rgb_feat, rgb_fuse)
                    for rgb_feat, rgb_fuse in zip(rgb_feats, rgb_fuses)
                ]
        else:
            rgb_fuses = rgb_feats

        return rgb_fuses, ir_feats

    def build_SwinTransformer(self):
        rgb_swin=SwinTransformer(embed_dims=self.embed_dims,
                                 in_channels=self.in_channels,
                                 depths=self.depths,
                                 num_heads=self.num_heads,
                                 window_size=self.window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=self.qkv_bias,
                                 drop_path_rate=self.drop_path_rate,
                                 out_indices=self.out_indices,
                                 init_cfg=self.init_cfg)
        ir_swin=SwinTransformer(embed_dims=self.embed_dims,
                                in_channels=self.in_channels,
                                 depths=self.depths,
                                 num_heads=self.num_heads,
                                 window_size=self.window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=self.qkv_bias,
                                 drop_path_rate=self.drop_path_rate,
                                 out_indices=self.out_indices,
                                 init_cfg=self.init_cfg)
        return rgb_swin,ir_swin

class Dual_Mode_Attention(nn.Module):
    """
    Deformable-style RGB-T fusion.

    Four feature levels are flattened and concatenated as in Deformable DETR /
    DINO, but each RGB query only samples a small number of keys from every
    level instead of attending to the whole memory.
    """

    def __init__(self, embed_dims, depths, num_heads=8, dropout=0.1,
                 attn_dim=256, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dim = attn_dim
        self.num_levels = len(depths)
        self.num_points = num_points
        self.in_channels = [self.embed_dims * (2 ** i) for i in range(self.num_levels)]

        assert self.attn_dim % self.num_heads == 0, \
            f'attn_dim={self.attn_dim} must be divisible by num_heads={self.num_heads}'
        self.head_dim = self.attn_dim // self.num_heads

        self.rgb_input_proj = nn.ModuleList([
            nn.Conv2d(in_channels, self.attn_dim, kernel_size=1)
            for in_channels in self.in_channels
        ])
        self.ir_input_proj = nn.ModuleList([
            nn.Conv2d(in_channels, self.attn_dim, kernel_size=1)
            for in_channels in self.in_channels
        ])
        self.output_proj_levels = nn.ModuleList([
            nn.Conv2d(self.attn_dim, in_channels, kernel_size=1)
            for in_channels in self.in_channels
        ])

        self.level_embed = nn.Parameter(torch.Tensor(self.num_levels, self.attn_dim))
        self.q_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.k_rgb_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.v_rgb_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.k_ir_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.v_ir_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.sampling_offsets = nn.Linear(
            self.attn_dim, self.num_heads * self.num_levels * self.num_points * 2)
        self.out_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.norm = nn.LayerNorm(self.attn_dim)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed)
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)

        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)

    def forward(self, rgb_feats: List[Tensor], ir_feats: List[Tensor]):
        assert len(rgb_feats) == self.num_levels and len(ir_feats) == self.num_levels

        rgb_proj_feats = [
            proj(feat) for proj, feat in zip(self.rgb_input_proj, rgb_feats)
        ]
        ir_proj_feats = [
            proj(feat) for proj, feat in zip(self.ir_input_proj, ir_feats)
        ]

        rgb_memory, spatial_shapes, level_start_index = self._flatten_mlvl_feats(
            rgb_proj_feats, add_level_embed=False)
        rgb_query_memory, _, _ = self._flatten_mlvl_feats(
            rgb_proj_feats, add_level_embed=True)
        B, L, _ = rgb_memory.shape
        device = rgb_memory.device
        valid_ratios = rgb_memory.new_ones(B, self.num_levels, 2)
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=device)

        query = self.q_proj(rgb_query_memory)
        sampling_offsets = self.sampling_offsets(query).view(
            B, L, self.num_heads, self.num_levels, self.num_points, 2)
        offset_normalizer = torch.stack(
            [spatial_shapes[:, 1], spatial_shapes[:, 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + \
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        k_rgb_feats, v_rgb_feats = self._build_kv_feats(
            rgb_proj_feats, self.k_rgb_proj, self.v_rgb_proj)
        k_ir_feats, v_ir_feats = self._build_kv_feats(
            ir_proj_feats, self.k_ir_proj, self.v_ir_proj)

        q = query.view(B, L, self.num_heads, self.head_dim)
        k_rgb = self._sample_head_features(k_rgb_feats, sampling_locations)
        v_rgb = self._sample_head_features(v_rgb_feats, sampling_locations)
        k_ir = self._sample_head_features(k_ir_feats, sampling_locations)
        v_ir = self._sample_head_features(v_ir_feats, sampling_locations)

        logits_rgb = (q[:, :, :, None, None, :] * k_rgb).sum(-1) / math.sqrt(self.head_dim)
        logits_ir = (q[:, :, :, None, None, :] * k_ir).sum(-1) / math.sqrt(self.head_dim)
        weight_rgb = F.softmax(logits_rgb.flatten(3), dim=-1).view_as(logits_rgb)
        weight_ir = F.softmax(logits_ir.flatten(3), dim=-1).view_as(logits_ir)

        out_rgb = (weight_rgb[..., None] * v_rgb).sum(dim=(3, 4))
        out_ir = (weight_ir[..., None] * v_ir).sum(dim=(3, 4))
        score_rgb = logits_rgb.flatten(3).max(dim=-1).values
        score_ir = logits_ir.flatten(3).max(dim=-1).values
        out = torch.where((score_rgb >= score_ir)[..., None], out_rgb, out_ir)

        out = out.reshape(B, L, self.attn_dim)
        out = self.out_proj(out)
        out = self.norm(rgb_memory + self.drop(out))
        return self._split_mlvl_feats(out, spatial_shapes, level_start_index)

    def _flatten_mlvl_feats(self, feats: List[Tensor], add_level_embed: bool):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(feats):
            B, C, H, W = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            feat = feat.flatten(2).permute(0, 2, 1)
            if add_level_embed:
                feat = feat + self.level_embed[lvl].view(1, 1, -1)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)

        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        return feat_flatten, spatial_shapes, level_start_index

    def _build_kv_feats(self, feats: List[Tensor], k_proj: nn.Linear,
                        v_proj: nn.Linear):
        k_feats, v_feats = [], []
        for lvl, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat = feat.flatten(2).permute(0, 2, 1)
            feat = feat + self.level_embed[lvl].view(1, 1, -1)
            k = k_proj(feat).view(B, H, W, self.attn_dim).permute(0, 3, 1, 2).contiguous()
            v = v_proj(feat).view(B, H, W, self.attn_dim).permute(0, 3, 1, 2).contiguous()
            k_feats.append(k)
            v_feats.append(v)
        return k_feats, v_feats

    def _sample_head_features(self, feats: List[Tensor], sampling_locations: Tensor):
        sampled_per_level = []
        B, L = sampling_locations.shape[:2]

        for lvl, feat in enumerate(feats):
            _, _, H, W = feat.shape
            grid = sampling_locations[:, :, :, lvl]
            grid = grid.mul(2.0).sub(1.0)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(
                B * self.num_heads, L, self.num_points, 2)

            feat = feat.view(B, self.num_heads, self.head_dim, H, W).reshape(
                B * self.num_heads, self.head_dim, H, W)
            sampled = F.grid_sample(
                feat, grid, mode='bilinear', padding_mode='zeros',
                align_corners=False)
            sampled = sampled.view(
                B, self.num_heads, self.head_dim, L, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()
            sampled_per_level.append(sampled)

        return torch.stack(sampled_per_level, dim=3)

    def _split_mlvl_feats(self, memory: Tensor, spatial_shapes: Tensor,
                          level_start_index: Tensor):
        outs = []
        for lvl, feat_proj in enumerate(self.output_proj_levels):
            H = int(spatial_shapes[lvl, 0].item())
            W = int(spatial_shapes[lvl, 1].item())
            start = int(level_start_index[lvl].item())
            length = H * W
            feat = memory[:, start:start + length]
            feat = feat.permute(0, 2, 1).reshape(
                memory.shape[0], self.attn_dim, H, W)
            outs.append(feat_proj(feat))
        return outs

    @staticmethod
    def get_encoder_reference_points(spatial_shapes: Tensor,
                                     valid_ratios: Tensor,
                                     device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class ComplementNet(nn.Module):
    def __init__(self,embed_dims, depths, dropout=0.1):
        super(ComplementNet, self).__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.norm_rgb = nn.ModuleList()
        self.norm_ir = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(len(depths)):
            self.norm_rgb.append(
                nn.LayerNorm(self.embed_dims * (2 ** i))
            )
            self.norm_ir.append(
                nn.LayerNorm(self.embed_dims * (2 ** i))
            )
            self.proj.append(
                nn.Linear(self.embed_dims * (2 ** i), self.embed_dims * (2 ** i))
            )

    def forward(self, rgb_feature, ir_feature, idx):
        """
        rgb_feature, ir_feature: [B, C, H, W]
        idx: 当前层索引
        """
        B, C, H, W = rgb_feature.shape

        add_feature = rgb_feature + ir_feature
        mul_feature = rgb_feature * ir_feature

        def normalize_feature(x):
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            return (x - mean) / std

        add_feature = normalize_feature(add_feature)
        mul_feature = normalize_feature(mul_feature)

        end_feature = add_feature - mul_feature
        end_feature = normalize_feature(end_feature)

        #可视化部分
        # view(rgb_feature, 'orgin_rgb')
        # view(ir_feature, 'orgin_ir')
        # view(add_feature, 'add_feature')
        # view(mul_feature, 'mul_feature')
        # view(end_feature, 'end_feature')

        end_feature = end_feature.permute(0, 2, 3, 1)  # [B, H, W, C]
        res = self.proj[idx](end_feature)
        res = self.dropout(res)

        rgb_feature = rgb_feature.permute(0, 2, 3, 1)
        ir_feature = ir_feature.permute(0, 2, 3, 1)

        # rgb_feature = rgb_feature + end_feature
        # ir_feature = ir_feature + end_feature
        rgb_feature = rgb_feature + res
        ir_feature = ir_feature + res


        rgb_feature = self.norm_rgb[idx](rgb_feature)
        ir_feature = self.norm_ir[idx](ir_feature)

        rgb_feature = rgb_feature.permute(0, 3, 1, 2)
        ir_feature = ir_feature.permute(0, 3, 1, 2)

        #可视化部分
        # view(rgb_feature, 'rgb_feature')
        # view(ir_feature, 'ir_feature')
        #
        # cv2.waitKey(0)
        # sys.exit(0)

        return rgb_feature, ir_feature

def view(image:Tensor,name:str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 300, 300)
    temp = image[0, 15].detach().cpu().numpy()
    temp = np.uint8((temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-8) * 255)
    cv2.imshow(name, temp)

def load_swin_to_submodule(submodule, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    # 如果 state_dict 的所有 key 以 'backbone.' 开头，去掉它
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith('backbone.'):
            nk = k[len('backbone.'):]
        new_state[nk] = v
    missing, unexpected = submodule.load_state_dict(new_state, strict=False)
    print('missing keys:', missing)
    print('unexpected keys:', unexpected)
    return missing, unexpected



