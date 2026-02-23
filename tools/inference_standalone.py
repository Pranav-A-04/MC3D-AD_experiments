"""
MC3D-AD Standalone Inference Script
Run anomaly detection on a single point cloud file WITHOUT requiring a dataset directory.

This script bypasses the template-based registration step, which means:
- You don't need the full dataset structure
- You don't need to specify a class name
- Results may be slightly different from the paper (registration helps alignment)

Usage:
    python tools/inference_standalone.py \
        --pcd_path /path/to/your/sample.pcd \
        --checkpoint ./experiments/real3d/checkpoints/ckpt_best.pth.tar \
        --pointmae_ckpt ./pretrain_ckp/modelnet_8k.pth \
        --visualize
"""

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Point-MAE Backbone Components (without registration)
# ============================================================================

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                point_cloud = ref[bi].detach().cpu()
                sample_points = query[bi].detach().cpu()
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(point_cloud.float(), point_cloud.float())
                distances, indices = knn.kneighbors(sample_points, n_neighbors=self.k)
                D.append(distances)
                I.append(indices)
            D = torch.from_numpy(np.array(D))
            I = torch.from_numpy(np.array(I))
        return D, I


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center, center_idx = fps(xyz.contiguous().float(), self.num_group)
        _, idx = self.knn(xyz, center)
        idx = idx.to(device=xyz.device)
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PointTransformerStandalone(nn.Module):
    """Point-MAE backbone WITHOUT registration - for standalone inference."""
    
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 40)
        )

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            self.load_state_dict(base_ckpt, strict=False)

    def forward(self, pointcloud):
        """Forward pass WITHOUT registration."""
        xyz = pointcloud["pointcloud"]  # (B, N, 3)
        B, N, _ = xyz.shape
        
        # Convert to (B, 3, N) then back - matching original flow
        xyz_input = xyz.permute(0, 2, 1).contiguous().float()  # (B, 3, N)
        
        if self.encoder_dims != self.trans_dim:
            pts = xyz_input.transpose(-1, -2)  # B N 3
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
            group_input_tokens = self.encoder(neighborhood)
            group_input_tokens = self.reduce_dim(group_input_tokens)
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
            pos = self.pos_embed(center)
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:, 1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)
        else:
            pts = xyz_input.transpose(-1, -2)
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
            group_input_tokens = self.encoder(neighborhood)
            pos = self.pos_embed(center)
            x = group_input_tokens
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(feat).transpose(-1, -2).contiguous() for feat in feature_list]
            x = feature_list[0]
        
        return {'xyz_features': x, 'center': center, 'ori_idx': ori_idx, 'center_idx': center_idx}


# ============================================================================
# UniAD Reconstruction Head
# ============================================================================

from sklearn.neighbors import KDTree
import copy
from typing import Optional
from torch import Tensor
from einops import rearrange


def flip_normals_to_outward(points, normals):
    centroid = np.mean(points, axis=0)
    directions = points - centroid
    dot_products = np.sum(normals * directions, axis=1)
    normals[dot_products < 0] = -normals[dot_products < 0]
    return normals


def estimate_parameters_kdtree(points, k=7, sample_size=1000):
    num_points = points.shape[0]
    sample_size = min(sample_size, num_points)
    kdtree = KDTree(points)
    sampled_indices = np.random.choice(num_points, sample_size, replace=False)
    sampled_points = points[sampled_indices]
    distances, _ = kdtree.query(sampled_points, k=2)
    avg_distance = np.mean(distances[:, 1])
    radius = k * avg_distance
    return radius, min(50, max(20, int(num_points / 1000)))


def analyze_normals_curvatures_optimized(point_cloud, k=5, radius=None):
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    B, N, _ = point_cloud.shape
    normals = np.zeros((B, N, 3))
    curvatures = np.zeros((B, N))
    normal_variations = np.zeros((B, N))
    curvature_variations = np.zeros((B, N))

    for b in range(B):
        group = point_cloud[b]
        kdtree = KDTree(group)
        if radius is None:
            radius, _ = estimate_parameters_kdtree(group)
        for i in range(N):
            idx = kdtree.query_radius(group[i].reshape(1, -1), r=radius)[0]
            neighbors = group[idx]
            if len(neighbors) < 3:
                continue
            centroid = np.mean(neighbors, axis=0)
            cov_matrix = np.cov((neighbors - centroid).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normals[b, i] = eigenvectors[:, 0]
            curvatures[b, i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)
            normals[b] = flip_normals_to_outward(group, normals[b])
        for i in range(N):
            idx = kdtree.query_radius(group[i].reshape(1, -1), r=radius)[0]
            neighbor_normals = normals[b, idx]
            dot_products = np.dot(neighbor_normals, normals[b, i])
            angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
            normal_variations[b, i] = np.mean(angles)
        for i in range(N):
            idx = kdtree.query_radius(group[i].reshape(1, -1), r=radius)[0]
            neighbor_curvatures = curvatures[b, idx]
            curvature_variations[b, i] = np.mean(np.abs(curvatures[b, i] - neighbor_curvatures))
    return normals, curvatures, normal_variations, curvature_variations


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayerUniAD(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderUniAD(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayerUniAD(nn.Module):
    def __init__(self, hidden_dim, feature_size, nhead, dim_feedforward, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.learned_embed = nn.Embedding(feature_size, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, out, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        tgt = pos
        tgt2 = self.self_attn(query=tgt, key=self.with_pos_embed(memory, pos), value=memory, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), key=self.with_pos_embed(out, pos), value=out, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderUniAD(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        output = memory
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerUniAD(nn.Module):
    def __init__(self, hidden_dim, feature_size, neighbor_mask, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask
        encoder_layer = TransformerEncoderLayerUniAD(hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoderUniAD(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayerUniAD(hidden_dim, feature_size, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoderUniAD(decoder_layer, num_decoder_layers, decoder_norm)
        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def generate_mask(self, feature_size, geome_vars, mask_ratio=0.4):
        B, _ = geome_vars.shape
        mask = torch.zeros((B, feature_size,), dtype=torch.bool)
        for idx in range(B):
            number = np.random.rand()
            if number > 0.5:
                sorted_indices_desc = np.argsort(geome_vars[idx])[::-1]
            else:
                sorted_indices_desc = np.argsort(geome_vars[idx])
            top_k = int(feature_size * mask_ratio)
            sorted_indices_desc = sorted_indices_desc[:top_k].copy()
            top_40_desc = torch.from_numpy(sorted_indices_desc).cuda()
            mask[idx, top_40_desc] = True
        return mask.cuda()

    def forward(self, src, pos_embed, geome_vars, mask_ratio):
        if self.neighbor_mask:
            mask = self.generate_mask(self.feature_size, geome_vars, mask_ratio)
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
        else:
            mask_enc = mask_dec1 = None
        output_encoder = self.encoder(src, src_key_padding_mask=mask_enc, pos=pos_embed)
        output_decoder = self.decoder(output_encoder, tgt_key_padding_mask=mask_dec1, pos=pos_embed)
        return output_decoder, output_encoder


class UniADStandalone(nn.Module):
    def __init__(self, feature_size, feature_jitter, neighbor_mask, hidden_dim, cls_num, inplanes=384, k=5, mask_ratio=0.4, **kwargs):
        super().__init__()
        self.feature_jitter = feature_jitter
        self.cls_num = cls_num
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, hidden_dim))
        self.transformer = TransformerUniAD(hidden_dim, feature_size, neighbor_mask, **kwargs)
        self.input_proj = nn.Linear(inplanes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(inplanes * 2, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, self.cls_num)
        )
        self.gem_dict = {}
        self.k = k
        self.mask_ratio = mask_ratio

    def forward(self, input_dict):
        feature_align = input_dict["xyz_features"]
        center = input_dict["center"]
        filename = input_dict["filename"][0] if isinstance(input_dict["filename"], list) else input_dict["filename"]
        
        if filename in self.gem_dict:
            geome_vars = self.gem_dict[filename]
        else:
            normals, curvatures, normal_variations, curvature_variations = analyze_normals_curvatures_optimized(center, k=self.k)
            geome_vars = normal_variations + 10 * curvature_variations
            self.gem_dict[filename] = geome_vars

        feature_tokens = rearrange(feature_align, "b n g -> g b n")
        feature_tokens = self.input_proj(feature_tokens)
        pos_embed = self.pos_embed(center).permute(1, 0, 2)
        output_decoder, _ = self.transformer(feature_tokens, pos_embed, geome_vars, self.mask_ratio)
        feature_rec_tokens = self.output_proj(output_decoder)
        feature_rec = rearrange(feature_rec_tokens, "g b n -> b n g")

        feature_cls = feature_rec.detach().clone()
        feature_cls.requires_grad = True
        feature_cls = rearrange(feature_cls, "b n g -> b g n")
        concat_f = torch.cat([feature_cls[:, 0], feature_cls[:, 1:].max(1)[0]], dim=-1)
        cls_pred = self.cls_head_finetune(concat_f)

        pred = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True))
        return {"feature_rec": feature_rec, "feature_align": feature_align, "pred": pred, "cls_pred": cls_pred}


# ============================================================================
# Standalone Model Wrapper
# ============================================================================

class MC3DADStandalone(nn.Module):
    """Complete MC3D-AD model for standalone inference (no registration)."""
    
    def __init__(self, pointmae_ckpt, num_group=1024, group_size=128, cls_num=40):
        super().__init__()
        self.backbone = PointTransformerStandalone(group_size=group_size, num_group=num_group, encoder_dims=384)
        self.backbone.load_model_from_ckpt(pointmae_ckpt)
        
        self.reconstruction = UniADStandalone(
            feature_size=num_group,
            feature_jitter={'scale': 20.0, 'prob': 1.0},
            neighbor_mask=type('obj', (object,), {'mask': [True, True, True]})(),
            hidden_dim=256,
            cls_num=cls_num,
            inplanes=384,
            k=5,
            mask_ratio=0.4,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            normalize_before=False
        )

    def forward(self, input_dict):
        backbone_out = self.backbone(input_dict)
        input_dict.update(backbone_out)
        recon_out = self.reconstruction(input_dict)
        input_dict.update(recon_out)
        return input_dict


# ============================================================================
# Inference Functions
# ============================================================================

def norm_pcd(point_cloud):
    center = np.average(point_cloud, axis=0)
    return point_cloud - np.expand_dims(center, axis=0)


def load_point_cloud(pcd_path):
    ext = os.path.splitext(pcd_path)[1].lower()
    if ext in ['.pcd', '.ply']:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.array(pcd.points)
    elif ext in ['.xyz', '.pts', '.txt']:
        try:
            points = np.loadtxt(pcd_path, delimiter=' ')[:, :3]
        except:
            points = np.loadtxt(pcd_path, delimiter=',')[:, :3]
    elif ext == '.npy':
        points = np.load(pcd_path)[:, :3]
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return points.astype(np.float32)


def fill_missing_values(x_data, x_label, y_data, k=1):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)
    _, indices = nn.kneighbors(y_data)
    return np.mean(x_label[indices], axis=1)


def save_pcd_with_scores(points, scores, output_path):
    """
    Save point cloud with anomaly scores as a colored PCD file.
    Points are colored from blue (low score) to red (high score).
    """
    # Normalize scores to [0, 1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Create color map: blue (low) -> yellow -> red (high)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = scores_norm  # Red channel
    colors[:, 1] = 1 - np.abs(scores_norm - 0.5) * 2  # Green channel (peaks at 0.5)
    colors[:, 2] = 1 - scores_norm  # Blue channel
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved colored point cloud to: {output_path}")


def run_inference(pcd_path, checkpoint_path, pointmae_ckpt, num_group=1024, device='cuda', output_pcd=None):
    """Run inference on a single PCD file."""
    
    total_start_time = time.time()
    
    # Load point cloud
    load_start = time.time()
    points_raw = load_point_cloud(pcd_path)
    points_normalized = norm_pcd(points_raw)
    pointcloud = torch.from_numpy(points_normalized).float()
    load_time = time.time() - load_start
    
    # Build model
    model_start = time.time()
    print("Building model...")
    model = MC3DADStandalone(pointmae_ckpt=pointmae_ckpt, num_group=num_group)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded!")
    model_load_time = time.time() - model_start
    
    # Prepare input
    input_dict = {
        'pointcloud': pointcloud.unsqueeze(0).to(device),
        'filename': [pcd_path],
        'label': torch.tensor([0]),
        'mask': torch.zeros(pointcloud.shape[0]),
    }
    
    # Run inference
    inference_start = time.time()
    if device == 'cuda':
        torch.cuda.synchronize()
    
    with torch.no_grad():
        outputs = model(input_dict)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    
    # Extract results
    postprocess_start = time.time()
    pred = outputs['pred'].squeeze(1).cpu().numpy()
    center_idx = outputs['center_idx'].cpu().numpy()
    point_cloud_out = outputs['pointcloud'].cpu().numpy()
    cls_pred = torch.argmax(outputs['cls_pred'], dim=1).cpu().numpy()
    
    # Interpolate to all points
    mask_idx = center_idx.squeeze().astype(np.int64)
    xyz_sampled = point_cloud_out[0][mask_idx, :]
    anomaly_scores = fill_missing_values(xyz_sampled, pred[0], point_cloud_out[0])
    
    # Smooth
    kernel_size = min(511, len(anomaly_scores) // 2 * 2 + 1)
    if kernel_size > 1:
        anomaly_scores = F.avg_pool1d(
            torch.from_numpy(anomaly_scores).unsqueeze(0).unsqueeze(0).float(),
            kernel_size=kernel_size, padding=kernel_size // 2, stride=1
        ).squeeze().numpy()
    postprocess_time = time.time() - postprocess_start
    
    # Save output PCD with anomaly scores as colors
    save_start = time.time()
    if output_pcd is None:
        # Default: save next to input file with _anomaly suffix
        base_name = os.path.splitext(pcd_path)[0]
        output_pcd = f"{base_name}_anomaly.pcd"
    
    save_pcd_with_scores(point_cloud_out[0], anomaly_scores, output_pcd)
    save_time = time.time() - save_start
    
    total_time = time.time() - total_start_time
    
    return {
        'anomaly_scores': anomaly_scores,
        'image_score': float(anomaly_scores.max()),
        'class_prediction': int(cls_pred[0]),
        'pointcloud': point_cloud_out[0],
        'num_points': len(anomaly_scores),
        'output_pcd': output_pcd,
        'timing': {
            'load_pcd': load_time,
            'model_load': model_load_time,
            'inference': inference_time,
            'postprocess': postprocess_time,
            'save_pcd': save_time,
            'total': total_time
        }
    }


def visualize_results(result, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return
    
    points = result['pointcloud']
    scores = result['anomaly_scores']
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, alpha=0.5)
    ax1.set_title('Point Cloud')
    
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=scores_norm, cmap='hot', s=1)
    ax2.set_title(f'Anomaly (max: {result["image_score"]:.4f})')
    plt.colorbar(scatter, ax=ax2, shrink=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MC3D-AD Standalone Inference')
    parser.add_argument('--pcd_path', type=str, required=True, help='Path to input .pcd file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to MC3D-AD checkpoint')
    parser.add_argument('--pointmae_ckpt', type=str, default='./pretrain_ckp/modelnet_8k.pth', help='Path to Point-MAE weights')
    parser.add_argument('--num_group', type=int, default=1024, help='Number of groups (1024 for ShapeNet, 4096 for Real3D)')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--threshold', type=float, default=None, help='Anomaly threshold')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--save_vis', type=str, default=None, help='Save visualization to file')
    parser.add_argument('--save_results', type=str, default=None, help='Save results to .npz')
    parser.add_argument('--output_pcd', type=str, default=None, 
                        help='Output PCD path (default: <input>_anomaly.pcd)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pcd_path):
        raise FileNotFoundError(f"PCD not found: {args.pcd_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.pointmae_ckpt):
        raise FileNotFoundError(f"Point-MAE weights not found: {args.pointmae_ckpt}")
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\nRunning inference on: {args.pcd_path}")
    result = run_inference(args.pcd_path, args.checkpoint, args.pointmae_ckpt, args.num_group, args.device, args.output_pcd)
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Number of points: {result['num_points']}")
    print(f"Anomaly score: {result['image_score']:.6f}")
    print(f"Score range: [{result['anomaly_scores'].min():.6f}, {result['anomaly_scores'].max():.6f}]")
    print(f"Output PCD: {result['output_pcd']}")
    
    if args.threshold is not None:
        is_anomaly = result['image_score'] > args.threshold
        print(f"Threshold: {args.threshold}")
        print(f"ANOMALY: {'YES' if is_anomaly else 'NO'}")
    
    print("-" * 50)
    print("TIMING")
    print("-" * 50)
    timing = result['timing']
    print(f"  Load PCD:     {timing['load_pcd']*1000:8.2f} ms")
    print(f"  Model load:   {timing['model_load']*1000:8.2f} ms")
    print(f"  Inference:    {timing['inference']*1000:8.2f} ms")
    print(f"  Postprocess:  {timing['postprocess']*1000:8.2f} ms")
    print(f"  Save PCD:     {timing['save_pcd']*1000:8.2f} ms")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:        {timing['total']*1000:8.2f} ms ({timing['total']:.2f} s)")
    print("=" * 50)
    
    if args.save_results:
        np.savez(args.save_results, **result)
        print(f"Results saved to: {args.save_results}")
    
    if args.visualize or args.save_vis:
        visualize_results(result, args.save_vis)


if __name__ == '__main__':
    main()
