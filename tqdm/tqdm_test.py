import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# 定义数据转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
])

# 下载并加载CIFAR10数据集
trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

from torch.utils.data import DataLoader

# 创建DataLoader
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# 使用DataLoader迭代数据
for images, labels in tqdm(trainloader):
    pass

for images, labels in tqdm(testloader):
    pass
