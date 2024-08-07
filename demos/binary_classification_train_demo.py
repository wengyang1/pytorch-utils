import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from datasets_utils.my_dataset_demo import MyDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 加载预训练的VGG16模型
vgg16 = models.vgg16(pretrained=True)

# 冻结特征提取层的参数
# for param in vgg16.features.parameters():
#     param.requires_grad = False

# 替换全连接层以匹配你的类别数
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 1)  # 替换为你自己的类别数

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = MyDataset('/Users/wengyang/datasets/hymenoptera_data', train=True, transform=transform)
val_dataset = MyDataset('/Users/wengyang/datasets/hymenoptera_data', train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.0001)

writer = SummaryWriter('../outputs')
# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    vgg16.train()
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = vgg16(inputs)
        outputs = outputs.view(-1, 1)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
        writer.add_scalar('Training/loss', loss.item(), epoch * len(train_loader) + i)
        loss.backward()
        optimizer.step()

print('Finished Training')

vgg16.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = vgg16(images)
        predicted = outputs > 0.9
        predicted = predicted.view(-1, 1)
        labels = labels.view(-1, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
