import os
from datetime import datetime

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else \
    (torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cpu'))

print('my device is {}'.format(DEVICE))

# 加载预训练的ResNet18模型
# model = models.resnet18(pretrained=True)
# 加载自己训练过的resnet18权重
resnet18 = models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features  # 获取全连接层的输入特征数
resnet18.fc = nn.Linear(num_ftrs, 10)  # 修改全连接层输出为10类
checkpoint = torch.load('../outputs/resnet18_cifar10/train20240814234139/best.pth', map_location='cpu')
resnet18.load_state_dict(checkpoint, strict=False)  # strict=False 忽略不匹配的key
resnet18.eval()  # 设置模型为评估模式

# 定义一个将PIL图像转换为模型输入张量的转换
transform_inference = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10的类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def inference_one(model, img):
    # 应用转换
    input_tensor = transform_inference(img).unsqueeze(0).to(DEVICE)  # 添加一个batch维度
    resnet18 = model.to(DEVICE)
    # 推理
    with torch.no_grad():
        output = resnet18(input_tensor)
    # 获取预测结果
    _, predicted = torch.max(output, 1)
    return predicted.item()


img_folder = '/Users/wengyang/datasets/cats'
output_path = os.path.join('../outputs', 'resnet18_cifar10', 'inference' + datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(os.path.join(output_path,'correct'),exist_ok=True)
os.makedirs(os.path.join(output_path,'wrong'),exist_ok=True)
img_num = len(os.listdir(img_folder))
correct_num = 0
for file in tqdm(os.listdir(img_folder)):
    true_value = 3
    img = cv2.imread(os.path.join(img_folder, file))
    output_file = os.path.join(output_path, file)
    pred = inference_one(resnet18, img)
    if pred == true_value:
        save_img_path = os.path.join(output_path, 'correct', file)
        correct_num += 1
    else:
        save_img_path = os.path.join(output_path, 'wrong', file)
        # 保存图片
    cv2.putText(img, f'Predicted: {classes[pred]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imwrite(save_img_path, img)
print('Accuracy on {} imgs is {}%'.format(img_num, round(100 * correct_num / img_num)))
