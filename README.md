# ncnn_depth_anything_cpp

## summary
Model conversion tool for depth_anything pnnx to the ncnn format.
An error occurred and it could not be converted correctly.
I am in the process of trying.

## result
```
$ python depth_anything_pytorch_export.py
$ pnnx depth_anything_vits14.pt inputshape=[1,3,518,518]
pnnxparam = depth_anything_vits14.pnnx.param
pnnxbin = depth_anything_vits14.pnnx.bin
pnnxpy = depth_anything_vits14_pnnx.py
pnnxonnx = depth_anything_vits14.pnnx.onnx
ncnnparam = depth_anything_vits14.ncnn.param
ncnnbin = depth_anything_vits14.ncnn.bin
ncnnpy = depth_anything_vits14_ncnn.py
fp16 = 1
optlevel = 2
device = cpu
inputshape = [1,3,518,518]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = depth_anything.blocks.FeatureFusionBlock
inline module = depth_anything.blocks.ResidualConvUnit
inline module = depth_anything.dpt.DPTHead
inline module = dinov2.layers.attention.MemEffAttention
inline module = dinov2.layers.block.NestedTensorBlock
inline module = dinov2.layers.layer_scale.LayerScale
inline module = dinov2.layers.mlp.Mlp
inline module = dinov2.layers.patch_embed.PatchEmbed
inline module = torch.nn.modules.linear.Identity
inline module = depth_anything.blocks.FeatureFusionBlock
inline module = depth_anything.blocks.ResidualConvUnit
inline module = depth_anything.dpt.DPTHead
inline module = dinov2.layers.attention.MemEffAttention
inline module = dinov2.layers.block.NestedTensorBlock
inline module = dinov2.layers.layer_scale.LayerScale
inline module = dinov2.layers.mlp.Mlp
inline module = dinov2.layers.patch_embed.PatchEmbed
inline module = torch.nn.modules.linear.Identity

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
force batch axis 233 for operand 23
force batch axis 233 for operand 29
force batch axis 233 for operand 46
force batch axis 233 for operand 52
force batch axis 233 for operand 69
force batch axis 233 for operand 75
force batch axis 233 for operand 92
force batch axis 233 for operand 98
force batch axis 233 for operand 115
force batch axis 233 for operand 121
force batch axis 233 for operand 138
force batch axis 233 for operand 144
force batch axis 233 for operand 161
force batch axis 233 for operand 167
force batch axis 233 for operand 184
force batch axis 233 for operand 190
force batch axis 233 for operand 207
force batch axis 233 for operand 213
force batch axis 233 for operand 230
force batch axis 233 for operand 236
force batch axis 233 for operand 253
force batch axis 233 for operand 259
force batch axis 233 for operand 276
force batch axis 233 for operand 282
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
binaryop broadcast across batch axis 0 and 233 is not supported
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
```
