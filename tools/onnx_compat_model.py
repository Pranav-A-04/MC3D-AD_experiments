import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# Import original backbone components
from tools.inference_standalone import PointTransformerStandalone, TransformerUniAD

class PurePyTorchKNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, ref, query):
        # ref: (B, N, 3), query: (B, G, 3)
        dist = torch.cdist(query, ref)
        val, idx = dist.topk(self.k, dim=-1, largest=False)
        return val, idx

def torch_generate_mask(feature_size, geome_vars, mask_ratio=0.4):
    """Replaces the NumPy masking logic with Torch for ONNX tracing."""
    # Ensure geome_vars is (B, N)
    if geome_vars.dim() == 3:
        geome_vars = geome_vars.squeeze(1)
        
    B = geome_vars.shape[0]
    k_val = int(feature_size * mask_ratio)
    
    # We use a fixed Top-K strategy for ONNX stability
    _, top_k_indices = torch.topk(geome_vars, k_val, dim=1)
    
    mask = torch.zeros((B, feature_size), dtype=torch.bool, device=geome_vars.device)
    mask.scatter_(1, top_k_indices, True)
    return mask

class MC3DAD_ONNX(nn.Module):
    def __init__(self, pointmae_ckpt, num_group=1024, group_size=128, cls_num=40):
        super().__init__()
        self.num_group = num_group
        
        # 1. Backbone
        self.backbone = PointTransformerStandalone(group_size=group_size, num_group=num_group, encoder_dims=384)
        
        # 2. Transformer
        self.transformer = TransformerUniAD(
            hidden_dim=256, feature_size=num_group, 
            neighbor_mask=type('obj', (object,), {'mask': [True, True, True]})(),
            nhead=8, num_encoder_layers=4, num_decoder_layers=4, 
            dim_feedforward=1024
        )
        # Apply the torch mask patch immediately
        self.transformer.generate_mask = torch_generate_mask
        
        # 3. Heads & Projections
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 256))
        self.input_proj = nn.Linear(384, 256)
        self.output_proj = nn.Linear(256, 384)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(384 * 2, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5),       # <-- Added back
            nn.Linear(256, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5),       # <-- Added back
            nn.Linear(256, cls_num)
        )

    def get_geometry_features(self, pcd, k=5):
        """Traceable curvature proxy."""
        B, N, _ = pcd.shape
        dist = torch.cdist(pcd, pcd)
        _, idx = dist.topk(k, dim=-1, largest=False)
        
        # Standardize neighbors
        neighbors = pcd.view(B, 1, N, 3).expand(-1, N, -1, -1)
        neighbors = torch.gather(neighbors, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        
        centroid = neighbors.mean(dim=2, keepdim=True)
        cov = torch.matmul((neighbors - centroid).transpose(-1, -2), (neighbors - centroid))
        trace = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        return trace / (trace.max(dim=-1, keepdim=True)[0] + 1e-8)

    def forward(self, pc):
        # Backbone logic
        backbone_out = self.backbone({"pointcloud": pc})
        feature_align = backbone_out["xyz_features"] # (B, 384, 1024)
        center = backbone_out["center"] # (B, 1024, 3)
        
        # Geometry (B, 1024)
        geome_vars = self.get_geometry_features(center)
        
        # Transformer (expects Seq, Batch, Dim)
        feature_tokens = rearrange(feature_align, "b d n -> n b d")
        feature_tokens = self.input_proj(feature_tokens)
        
        pos_embed_val = self.pos_embed(center).permute(1, 0, 2)
        
        # The mask is generated inside transformer.forward using our patched torch_generate_mask
        output_decoder, _ = self.transformer(feature_tokens, pos_embed_val, geome_vars, 0.4)
        
        feature_rec_tokens = self.output_proj(output_decoder)
        feature_rec = rearrange(feature_rec_tokens, "n b d -> b d n")
        
        # Anomaly Map Calculation
        anomaly_map = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1))
        
        # Class Pred
        concat_f = torch.cat([feature_rec[:, :, 0], feature_rec.max(dim=-1)[0]], dim=-1)
        cls_pred = self.cls_head_finetune(concat_f)
        
        return anomaly_map, cls_pred

def patch_model_for_onnx(model):
    """Replaces all KNN modules in the backbone with PurePyTorchKNN."""
    # Step 1: Collect all the modules we need to replace first
    targets = []
    for name, module in model.named_modules():
        if 'knn' in name.lower() and isinstance(module, nn.Module):
            if hasattr(module, 'k'):
                targets.append((name, module.k))
                
    # Step 2: Perform the replacement outside the generator loop
    for name, k_val in targets:
        parent_name = name.rsplit('.', 1)[0] if '.' in name else None
        attr_name = name.rsplit('.', 1)[-1]
        
        if parent_name:
            # Get the parent module
            parent = dict(model.named_modules())[parent_name]
            # Replace the child attribute with our PyTorch native KNN
            setattr(parent, attr_name, PurePyTorchKNN(k=k_val))
            print(f"Patched {name} -> PurePyTorchKNN(k={k_val})")
        else:
            setattr(model, name, PurePyTorchKNN(k=k_val))
            print(f"Patched {name} -> PurePyTorchKNN(k={k_val})")
            
    return model