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

"""
import ncnn  # noqa
# Export the PyTorch model to ONNX format
#torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"], verbose=True)
    
print(f"Model exported to ")
print(f'\n starting export with ncnn {ncnn.__version__}...')

#opt_model = pnnx.export(depth_anything, "depth_anything_vits_14_trial", dummy_input)


# 半精度浮動小数点数（FP16）を使用するかどうか（1は使用、0は不使用）
fp16_data_type = 0

# グラフ最適化レベル（0, 1, 2のいずれか）
opt_level = 2

# PNNXに渡す追加の引数
pnnx_args = [
    'pnnxparam=model.pnnx.param',
    'pnnxbin=model.pnnx.bin',
    'pnnxpy=model_pnnx.py',
    'pnnxonnx=model.pnnx.onnx',
]

# コマンドラインコマンドの構築
cmd = [
    "pnnx",
    traced_script_module_file,
    *pnnx_args,
    f"optlevel={opt_level}",
    f"fp16={fp16_data_type}",
#    f"device={'cpu'}",
    f'inputshape={[1,3,518,518]}',  # モデル入力形状を動的に設定
]

# コマンドの実行と結果の出力
print(f"Running: {' '.join(cmd)}")
ret = subprocess.run(cmd, check=True, capture_output=True, text=True)
print('Return code:', ret.returncode)
print('stdout:', ret.stdout)

# 不要なデバッグファイルの削除
pnnx_files = [arg.split('=')[-1] for arg in pnnx_args]
for f_debug in ["debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files]:
    file_path = Path(f_debug)
    if file_path.exists():
        file_path.unlink()
        print(f'Removed: {file_path}')
        
"""
            
