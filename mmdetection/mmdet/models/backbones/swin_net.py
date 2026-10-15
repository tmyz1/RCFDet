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
from torch import Tensor
from mmdet.models.necks import YOLOXPAFPN


@MODELS.register_module()
class SwinNet(BaseModule):
    def __init__(self,in_channels:int,embed_dims:int,depths:tuple[int],num_heads:tuple[int],window_size:int,
                 mlp_ratio:int,qkv_bias,drop_path_rate:float,out_indices:tuple[int],
                 pretrained=None,init_cfg=None,act_cfg=dict(type='GELU'),fusion_channels=256):
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
            with_img_mask=True,
            fusion_channels=fusion_channels,
        )

    def forward(self,x):
        rgb,ir,a=x
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
                 pretrained=None,init_cfg=None,act_cfg=dict(type='GELU'),with_img_mask=True,
                 fusion_channels=256):
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
        load_swin_to_submodule(self.rgb_swin, r'/root/autodl-tmp/weight/swin_tiny_224.pth')
        #load_swin_to_submodule(self.rgb_swin, r"D:\pycharm work\weight\swin_tiny_224.pth")
        self.ir_swin.load_state_dict(self.rgb_swin.state_dict(), strict=False)  # 复制给另一个分支

        stage_channels = [self.embed_dims * (2 ** i) for i in range(len(self.depths))]
        self.rgb_pafpn = YOLOXPAFPN(
            in_channels=stage_channels,
            out_channels=fusion_channels,
            num_csp_blocks=len(stage_channels),
        )
        self.ir_pafpn = YOLOXPAFPN(
            in_channels=stage_channels,
            out_channels=fusion_channels,
            num_csp_blocks=len(stage_channels),
        )
        #模态之间的注意力计算
        self.dual_attn = Dual_Mode_Attention(channels=fusion_channels, num_levels=2)

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
        if a.numel() != rgb_feats[0].shape[0] or not torch.isfinite(a).all():
            raise ValueError('a must contain one finite RGB weight per image')
        if ((a < 0) | (a > 1)).any():
            raise ValueError('RGB weights must lie in [0, 1]')
        dual_mask = (a > 0) & (a < 1)
        rgb_feats = list(self.rgb_pafpn(rgb_feats))
        ir_feats = list(self.ir_pafpn(ir_feats))

        if dual_mask.any():
            selected = dual_mask.nonzero(as_tuple=True)[0]
            rgb_selected = [feat.index_select(0, selected) for feat in rgb_feats[-2:]]
            ir_selected = [feat.index_select(0, selected) for feat in ir_feats[-2:]]
            rgb_cross, ir_cross = self.dual_attn(rgb_selected, ir_selected)
            rgb_strength = (1.0 - a.index_select(0, selected)).view(-1, 1, 1, 1)
            ir_strength = a.index_select(0, selected).view(-1, 1, 1, 1)
            for offset in range(2):
                level = len(rgb_feats) - 2 + offset
                rgb_updated = rgb_selected[offset] + rgb_strength * (
                    rgb_cross[offset] - rgb_selected[offset]
                )
                ir_updated = ir_selected[offset] + ir_strength * (
                    ir_cross[offset] - ir_selected[offset]
                )
                rgb_feats[level] = torch.index_copy(rgb_feats[level], 0, selected, rgb_updated)
                ir_feats[level] = torch.index_copy(ir_feats[level], 0, selected, ir_updated)

        return rgb_feats, ir_feats

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
    """Bidirectional cross-modal attention over the last two pyramid levels."""

    def __init__(self, channels=256, num_levels=2, num_heads=8,
                 dropout=0.1, num_points=4, max_offset=2.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError('channels must be divisible by num_heads')
        if num_points != 4 or num_levels != 2:
            raise ValueError('attention requires four corners on each of two levels')
        if max_offset <= 0:
            raise ValueError('max_offset must be positive')
        self.channels = channels
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.max_offset = max_offset
        self.head_dim = channels // num_heads

        self.rgb_input_proj = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1)
            for _ in range(num_levels)
        ])
        self.ir_input_proj = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1)
            for _ in range(num_levels)
        ])

        self.level_embed = nn.Parameter(torch.empty(num_levels, channels))
        self.q_rgb_proj = nn.Linear(channels, channels)
        self.q_ir_proj = nn.Linear(channels, channels)
        self.k_rgb_proj = nn.Linear(channels, channels)
        self.v_rgb_proj = nn.Linear(channels, channels)
        self.k_ir_proj = nn.Linear(channels, channels)
        self.v_ir_proj = nn.Linear(channels, channels)
        offset_dims = num_heads * num_levels * num_points * 2
        self.rgb_offsets = nn.Linear(channels, offset_dims)
        self.ir_offsets = nn.Linear(channels, offset_dims)
        self.rgb_out_proj = nn.Linear(channels, channels)
        self.ir_out_proj = nn.Linear(channels, channels)
        self.rgb_norm = nn.LayerNorm(channels)
        self.ir_norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.register_buffer(
            'corner_signs',
            torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]),
            persistent=False,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed, std=0.02)
        for layer in (self.rgb_offsets, self.ir_offsets):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, rgb_feats: List[Tensor], ir_feats: List[Tensor]):
        assert len(rgb_feats) == self.num_levels and len(ir_feats) == self.num_levels

        rgb_proj_feats = [
            proj(feat) for proj, feat in zip(self.rgb_input_proj, rgb_feats)
        ]
        ir_proj_feats = [
            proj(feat) for proj, feat in zip(self.ir_input_proj, ir_feats)
        ]
        rgb_cross = self._cross_attention(
            rgb_proj_feats, ir_proj_feats, self.q_rgb_proj,
            self.k_ir_proj, self.v_ir_proj, self.rgb_offsets,
            self.rgb_out_proj, self.rgb_norm,
        )
        ir_cross = self._cross_attention(
            ir_proj_feats, rgb_proj_feats, self.q_ir_proj,
            self.k_rgb_proj, self.v_rgb_proj, self.ir_offsets,
            self.ir_out_proj, self.ir_norm,
        )
        return rgb_cross, ir_cross

    def _cross_attention(self, target_feats, source_feats, q_proj, k_proj,
                         v_proj, offset_proj, out_proj, norm):
        target_memory, spatial_shapes, level_start_index = self._flatten_mlvl_feats(
            target_feats, add_level_embed=False)
        query_memory, _, _ = self._flatten_mlvl_feats(
            target_feats, add_level_embed=True)
        batch, length, _ = target_memory.shape
        valid_ratios = target_memory.new_ones(batch, self.num_levels, 2)
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=target_memory.device
        ).to(target_memory.dtype)
        query = q_proj(query_memory)
        locations = self._sampling_locations(
            query, reference_points, spatial_shapes, offset_proj
        )
        k_feats, v_feats = self._build_kv_feats(source_feats, k_proj, v_proj)
        keys = self._sample_head_features(k_feats, locations)
        values = self._sample_head_features(v_feats, locations)
        q = query.reshape(batch, length, self.num_heads, self.head_dim)
        logits = (q[:, :, :, None, None, :] * keys).sum(-1) / math.sqrt(self.head_dim)
        attention = F.softmax(logits.flatten(3), dim=-1).view_as(logits)
        attended = (attention[..., None] * values).sum(dim=(3, 4))
        attended = out_proj(attended.reshape(batch, length, self.channels))
        output = norm(target_memory + self.drop(attended))
        return self._split_mlvl_feats(output, spatial_shapes, level_start_index)

    def _sampling_locations(self, query, reference_points, spatial_shapes,
                            offset_proj):
        batch, length, _ = query.shape
        radius = torch.sigmoid(offset_proj(query)).view(
            batch, length, self.num_heads, self.num_levels, self.num_points, 2
        ) * self.max_offset
        normalizer = torch.stack(
            [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1
        ).to(query.dtype)
        center = reference_points[:, :, None, :, None, :]
        signs = self.corner_signs.to(query.dtype).view(1, 1, 1, 1, 4, 2)
        available = torch.where(signs < 0, center, 1.0 - center) * 0.99
        step = torch.minimum(
            radius / normalizer[None, None, None, :, None, :], available
        )
        return (center + signs * step).clamp(0.0, 1.0)

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
            k = k_proj(feat).view(B, H, W, self.channels).permute(0, 3, 1, 2).contiguous()
            v = v_proj(feat).view(B, H, W, self.channels).permute(0, 3, 1, 2).contiguous()
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
                feat, grid, mode='bilinear', padding_mode='border',
                align_corners=False)
            sampled = sampled.view(
                B, self.num_heads, self.head_dim, L, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()
            sampled_per_level.append(sampled)

        return torch.stack(sampled_per_level, dim=3)

    def _split_mlvl_feats(self, memory: Tensor, spatial_shapes: Tensor,
                          level_start_index: Tensor):
        outs = []
        for lvl in range(self.num_levels):
            H = int(spatial_shapes[lvl, 0].item())
            W = int(spatial_shapes[lvl, 1].item())
            start = int(level_start_index[lvl].item())
            length = H * W
            feat = memory[:, start:start + length]
            feat = feat.permute(0, 2, 1).reshape(
                memory.shape[0], self.channels, H, W)
            outs.append(feat)
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



