import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from thop import profile

'''
使用vgg16处理cifar10数据集的10分类任务
'''
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else \
    (torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cpu'))

print('my train device is {}'.format(DEVICE))


def cal_op(model):
    input = torch.randn(1, 3, 224, 224)  # 假设输入是一个批量的3通道224x224图像
    input = input.to(DEVICE)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9} G")  # 将FLOPs转换为以十亿次浮点运算为单位（GFLOPs）
    print(f"Params: {params / 1e6} M")  # 将参数量转换为以百万为单位（MParams）


def save_model(model, save_path):
    model_state_dict = model.state_dict()
    model_path = os.path.join(save_path, 'last.pth')
    torch.save(model_state_dict, model_path)
    print('save model to {}'.format(model_path))


def test_model(model, testloader, global_step, writer):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    writer.add_scalar('accuracy', correct/total, global_step)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')


# 加载预训练的vgg16模型
vgg16 = models.vgg16(pretrained=False)

# 替换全连接层以匹配你的类别数
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 10)  # 替换为你自己的类别数
vgg16.to(DEVICE)
# 根据模型参数所在的device判断是否利用了gpu
print(f"Model is running on {next(vgg16.parameters()).device}.")

cal_op(vgg16)

# 定义数据转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
])

# 下载并加载CIFAR10数据集
trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()

num_epochs = 50

optimizer = optim.Adam(vgg16.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(trainloader), eta_min=1e-5)

output_path = os.path.join('../outputs', 'vgg16_cifar10', datetime.now().strftime("%Y%m%d%H%M%S"))
writer = SummaryWriter(output_path)
# 训练模型
global_step = 0
for epoch in range(num_epochs):
    vgg16.train()
    epoch_iterator = tqdm(trainloader)
    for i, data in enumerate(epoch_iterator):
        global_step = epoch * len(trainloader) + i
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        epoch_iterator.set_description(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}],global_step {global_step}, Loss: {loss.item()}')
        if global_step % 20 == 0:
            writer.add_scalar('loss', loss.item(), global_step)
            writer.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    save_model(vgg16, output_path)
    test_model(vgg16, testloader, global_step, writer)
print('Finished Training')
