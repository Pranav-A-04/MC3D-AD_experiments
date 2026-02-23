import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# Note: Ensure the base components (Transformer, Encoder, etc.) 
# from your original script are available in the same directory.
from tools.inference_standalone import PointTransformerStandalone, TransformerUniAD

class MC3DAD_ONNX(nn.Module):
    def __init__(self, pointmae_ckpt, num_group=1024, group_size=128, cls_num=40):
        super().__init__()
        # 1. Backbone
        self.backbone = PointTransformerStandalone(group_size=group_size, num_group=num_group, encoder_dims=384)
        # We load the weights later in the script to ensure they match the .pt file
        
        # 2. Reconstruction Parameters
        self.num_group = num_group
        self.inplanes = 384
        self.hidden_dim = 256
        self.mask_ratio = 0.4
        
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.hidden_dim))
        self.input_proj = nn.Linear(self.inplanes, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.inplanes)
        
        # 3. Transformer
        self.transformer = TransformerUniAD(
            self.hidden_dim, num_group, 
            neighbor_mask=type('obj', (object,), {'mask': [True, True, True]})(),
            nhead=8, num_encoder_layers=4, num_decoder_layers=4, 
            dim_feedforward=1024
        )
        
        # 4. Classification Head
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.inplanes * 2, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, cls_num)
        )

    def get_geometry_features(self, pcd, k=5):
        """Pure PyTorch implementation of normal/curvature estimation for ONNX tracing."""
        # pcd: (B, N, 3)
        dist = torch.cdist(pcd, pcd)
        _, idx = dist.topk(k, largest=False) # (B, N, K)
        
        B, N, K = idx.shape
        # Gather neighbors: (B, N, K, 3)
        neighbors = pcd.view(B, 1, N, 3).expand(-1, N, -1, -1)
        neighbors = torch.gather(neighbors, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        
        centroid = neighbors.mean(dim=2, keepdim=True)
        centered = neighbors - centroid
        cov = torch.matmul(centered.transpose(-1, -2), centered) / (K - 1)
        
        # Approximate curvature using the trace/eigenvalue logic in a traceable way
        # For ONNX, we use a simplified variation since linalg.eigh can be unstable in export
        trace = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        # Heuristic for curvature variation
        curvature = trace / (trace.sum(dim=-1, keepdim=True) + 1e-8)
        return curvature # Simplified for ONNX stability

    def forward(self, pc):
        # Backbone Forward
        backbone_out = self.backbone({"pointcloud": pc})
        feature_align = backbone_out["xyz_features"]
        center = backbone_out["center"]
        
        # Geometry Analysis (Native PyTorch)
        geome_vars = self.get_geometry_features(center).unsqueeze(0)
        
        # Transformer logic
        feature_tokens = rearrange(feature_align, "b n g -> g b n")
        feature_tokens = self.input_proj(feature_tokens)
        pos_embed = self.pos_embed(center).permute(1, 0, 2)
        
        output_decoder, _ = self.transformer(feature_tokens, pos_embed, geome_vars, self.mask_ratio)
        feature_rec_tokens = self.output_proj(output_decoder)
        feature_rec = rearrange(feature_rec_tokens, "g b n -> b n g")
        
        # Prediction
        concat_f = torch.cat([feature_rec[:, 0], feature_rec[:, 1:].max(1)[0]], dim=-1)
        cls_pred = self.cls_head_finetune(concat_f)
        diff = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1))
        
        return diff, cls_pred