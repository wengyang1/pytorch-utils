import torch
from torchvision import models

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else \
    (torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cpu'))

print('my device is {}'.format(DEVICE))

vgg16 = models.vgg16(pretrained=True)
print(f"Model is running on {next(vgg16.parameters()).device}.")
# move model to device and check if success
vgg16 = vgg16.to(DEVICE)
print(f"Model is running on {next(vgg16.parameters()).device}.")
for k, v in vgg16.named_parameters():
    print('param name {} param shape {}'.format(k, v.shape))
