import re

import cv2
import numpy as np
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
        rgb_fuses,ir_fuses=self.CRFuse(rgb, ir)
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
        load_swin_to_submodule(self.rgb_swin, r'/root/autodl-tmp/Swin-Deformable/mmdetection/weight/swin_big_224.pth')
        #load_swin_to_submodule(self.rgb_swin, r"E:\reproduce\weights\SwinTransformer\swin_big_224.pth")
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

    def forward(self, rgb, ir):
        rgb_hw_shape = ir_hw_shape = None
        rgb_fuses:List[Tensor]=[]
        ir_fuses:List[Tensor]=[]
        for i in range(0, len(self.depths)):
            rgb, _, rgb_out, rgb_hw_shape,rgb_img_mask = self.rgb_swin.forward_stage(rgb, i, rgb_hw_shape)
            ir, _, ir_out, ir_hw_shape,ir_img_mask = self.ir_swin.forward_stage(ir, i, ir_hw_shape)
            #模态之间的注意力计算
            B, C, H, W = rgb_out.shape
            if i == len(self.depths) - 1:
                inter_R = self.dual_attn(rgb_out, ir_out, i)
                rgb_fuses.append(inter_R)
                inter_T = self.dual_attn(ir_out, rgb_out, i)
                ir_fuses.append(inter_T)
            else:
                inter_R = rgb_out
                rgb_fuses.append(inter_R)
                inter_T = ir_out
                ir_fuses.append(inter_T)
            #下采样
            if i is not len(self.depths) - 1:
                B, C, H, W = inter_R.shape
                inter_R = inter_R.flatten(2).permute(0, 2, 1)
                inter_T = inter_T.flatten(2).permute(0, 2, 1)
                rgb, rgb_hw_shape = self.downsample[i](inter_R, rgb_hw_shape)
                rgb = rgb.view(B, -1, H // 2, W // 2)
                rgb = rgb.flatten(2).permute(0, 2, 1)
                ir, ir_hw_shape = self.downsample[i](inter_T, ir_hw_shape)
                ir = ir.view(B, -1, H // 2, W // 2)
                ir = ir.flatten(2).permute(0, 2, 1)
            #上下层融合

        return rgb_fuses,ir_fuses,

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
    双模态多头注意力模块
    """
    def __init__(self, embed_dims, depths, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.dropout = dropout

        self.q_main = nn.ModuleList()
        self.k_main = nn.ModuleList()
        self.v_main = nn.ModuleList()
        self.k_aux = nn.ModuleList()
        self.v_aux = nn.ModuleList()
        self.out_proj = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for i in range(len(self.depths)):
            dim = self.embed_dims * (2 ** i)
            self.q_main.append(nn.Linear(dim, dim))
            self.k_main.append(nn.Linear(dim, dim))
            self.v_main.append(nn.Linear(dim, dim))
            self.k_aux.append(nn.Linear(dim, dim))
            self.v_aux.append(nn.Linear(dim, dim))
            self.out_proj.append(nn.Linear(dim, dim))
            self.norm.append(nn.LayerNorm(dim))

    def forward(self, main, aux, idx):
        B, C, H, W = main.shape
        L = H * W
        head_dim = C // self.num_heads
        main = main.flatten(2).permute(0, 2, 1)  # [B, L, C]
        aux = aux.flatten(2).permute(0, 2, 1) # [B, L, C]
        # 位置编码
        pos_main = self.get_2d_sincos_pos_embed(embed_dim=C, grid_size=[H, W])
        pos_main = pos_main.to(main.device)
        pos_aux = pos_main
        main = main + pos_main.flatten(2).permute(0, 2, 1).expand(B, -1, -1)
        aux = aux + pos_aux.flatten(2).permute(0, 2, 1).expand(B, -1, -1)

        Q= self.q_main[idx](main)
        K_main = self.k_main[idx](main)
        V_main = self.v_main[idx](main)
        K_aux = self.k_aux[idx](aux)
        V_aux = self.v_aux[idx](aux)

        K = torch.cat([K_aux, K_main], dim=1)  # [B, 2L, C]
        V = torch.cat([V_aux, V_main], dim=1) # [B, 2L, C]

        def reshape_heads(x, Lx):
            return x.view(B, Lx, self.num_heads, head_dim).transpose(1, 2)

        Q = reshape_heads(Q, L)
        K = reshape_heads(K, 2 * L)
        V = reshape_heads(V, 2 * L)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # [B, num_heads, L, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj[idx](out)
        out = self.drop(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """二维正余弦位置编码"""
        H, W = grid_size
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        pos_x = x.flatten()
        pos_y = y.flatten()
        omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
        omega = 1.0 / (10000 ** omega)

        out_x = torch.einsum('m,d->md', pos_x, omega)
        out_y = torch.einsum('m,d->md', pos_y, omega)

        emb = torch.cat([
            torch.sin(out_x), torch.cos(out_x),
            torch.sin(out_y), torch.cos(out_y)
        ], dim=1)
        emb = emb.reshape(H, W, embed_dim).permute(2, 0, 1).unsqueeze(0)
        return emb


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



