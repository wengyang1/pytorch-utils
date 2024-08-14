import torch
import torch.nn as nn

# 创建模型输出和目标标签
inputs = torch.randn(10, 1, requires_grad=True)  # 假设有10个样本，每个样本的输出是一个实数
targets = torch.randint(2, (10, 1), dtype=torch.float)  # 目标标签，0或1
print(inputs.view(1, -1))
print(targets.view(1, -1))
# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 计算损失
loss = criterion(inputs, targets)

# 反向传播
loss.backward()
print(f"Loss: {loss.item()}")

print(f"Gradients: {inputs.grad.view(1, -1)}")