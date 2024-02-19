from pathlib import Path
import subprocess
import torch
import torch.onnx

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

encoder = 'vits'
load_from = f'./checkpoints/depth_anything_{encoder}14.pth'
image_shape = (3, 518, 518)

# Initializing model
assert encoder in ['vits', 'vitb', 'vitl']
if encoder == 'vits':
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub='localhub')
elif encoder == 'vitb':
    depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub='localhub')
else:
    depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub='localhub')

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

# Loading model weight
depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'), strict=True)
depth_anything.eval()

# Define dummy input data
dummy_input = torch.ones(image_shape).unsqueeze(0)

# Provide an example input to the model, this is necessary for exporting to ONNX
example_output = depth_anything(dummy_input)

onnx_path = load_from.split('/')[-1].split('.pth')[0] + '.onnx'

traced_script_module = torch.jit.trace(depth_anything, dummy_input, strict=True)
traced_script_module_file="depth_anything_vits14.pt"
traced_script_module.save(traced_script_module_file)

        print(f'Removed: {file_path}')
        
"""
            
