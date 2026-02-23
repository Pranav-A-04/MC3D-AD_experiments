import torch
import torch.onnx
from tools.onnx_compat_model import MC3DAD_ONNX

def export():
    # Configuration
    CKPT_PATH = "./experiments/real3d/checkpoints/ckpt_best.pth.tar"
    PMAE_PATH = "./pretrain_ckp/modelnet_8k.pth"
    OUTPUT_ONNX = "mc3dad_v1.onnx"
    NUM_POINTS = 8192 # Adjust based on your typical input
    
    # 1. Instantiate Model
    model = MC3DAD_ONNX(pointmae_ckpt=PMAE_PATH)
    
    # 2. Load Weights
    print(f"Loading checkpoint from {CKPT_PATH}...")
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Clean state dict keys
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. Create Dummy Input
    dummy_pc = torch.randn(1, NUM_POINTS, 3)

    # 4. Export
    print("Starting ONNX export (this may take a few minutes)...")
    torch.onnx.export(
        model,
        dummy_pc,
        OUTPUT_ONNX,
        export_params=True,
        opset_version=14, # Higher opset for better 3D op support
        do_constant_folding=True,
        input_names=['point_cloud'],
        output_names=['anomaly_map', 'class_logits'],
        dynamic_axes={
            'point_cloud': {1: 'num_points'},
            'anomaly_map': {1: 'num_points'}
        }
    )
    print(f"Export complete: {OUTPUT_ONNX}")

if __name__ == "__main__":
    export()