import torch
import torch.nn as nn
import torch.nn.functional as F


# 假设你有一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # 假设只有一个全连接层

    def forward(self, x):
        return self.fc(x)  # 直接返回logits


# 实例化模型
model = SimpleModel(input_size=10, num_classes=3)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 假设你有一些输入数据和真实标签
inputs = torch.randn(1, 10)  # 假设批次大小为1，输入特征维度为10
targets = torch.tensor([1])  # 假设真实标签是类别1（注意是整数索引）

# 前向传播得到logits
logits = model(inputs)

# 计算损失
loss = criterion(logits, targets)

print(loss)