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

'''
使用vgg16处理cifar10数据集的10分类任务
'''

def cal_op(model):
    print()


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else \
    (torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cpu'))

print('my train device is {}'.format(DEVICE))

# 加载预训练的VGG16模型
vgg16 = models.vgg16(pretrained=True)

# 替换全连接层以匹配你的类别数
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 10)  # 替换为你自己的类别数
vgg16.to(DEVICE)
# 根据模型参数所在的device判断是否利用了gpu
print(f"Model is running on {next(vgg16.parameters()).device}.")

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

num_epochs = 100

optimizer = optim.Adam(vgg16.classifier.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(trainloader), eta_min=1e-4)

output_path = os.path.join('../outputs', 'vgg16_cifar10', datetime.now().strftime("%Y%m%d%H%M%S"))
writer = SummaryWriter(output_path)
# 训练模型
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
    model_state_dict = vgg16.state_dict()
    model_path = os.path.join(output_path, 'last.pth')
    torch.save(model_state_dict, model_path)
    print('save model to {}'.format(model_path))
print('Finished Training')
