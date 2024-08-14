from torchvision import models
import torch
from thop import profile
vgg16 = models.vgg16(pretrained=True)

'''
注意 input的batch,img_size会影响到计算量
'''
input = torch.randn(1, 3, 224, 224)  # 假设输入是一个批量的3通道224x224图像
flops, params = profile(vgg16, inputs=(input,))
print(f"FLOPs: {flops / 1e9} G")  # 将FLOPs转换为以十亿次浮点运算为单位（GFLOPs）
print(f"Params: {params / 1e6} M")  # 将参数量转换为以百万为单位（MParams）

# 方法2:遍历param，统计参数量
num_params = 0
for p in vgg16.parameters():
    num_params += p.numel()
print(f"Params: {num_params / 1e6} M")  # 将参数量转换为以百万为单位（MParams）
