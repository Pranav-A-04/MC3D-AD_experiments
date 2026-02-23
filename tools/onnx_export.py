import torch
import os
import sys

# This dynamically finds the 'MC3D-AD_experiments' folder regardless of where you run the script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Insert it at the very top of Python's search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now it can safely find the 'tools' package
from tools.onnx_compat_model import MC3DAD_ONNX, patch_model_for_onnx

def export():
    CKPT_PATH = "/content/ckpt_best.pth.tar"
    PMAE_PATH = "/content/MC3D-AD_experiments/pretrain_ckp/modelnet_8k.pth"
    
    model = MC3DAD_ONNX(pointmae_ckpt=PMAE_PATH)
    
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    
    # Load state dict with 'reconstruction.' prefix removal
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('reconstruction.', '')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model = patch_model_for_onnx(model)
    model.eval()

    dummy_pc = torch.randn(1, 45000, 3)

    print("Exporting...")
    torch.onnx.export(
        model,
        dummy_pc,
        "mc3dad_final.onnx",
        export_params=True,
        opset_version=18,             # <-- UPGRADE THIS TO 18
        do_constant_folding=False,    # <-- SET THIS TO FALSE
        input_names=['point_cloud'],
        output_names=['anomaly_map', 'class_logits'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )
    print("Done!")

if __name__ == "__main__":
    export()