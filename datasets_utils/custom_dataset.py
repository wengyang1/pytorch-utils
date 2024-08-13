import os.path

import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2


def traverse_dir(directory):
    items = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if '.jpg' in name:
                items.append(os.path.join(root, name))
    return items


class MyDataset(Dataset):
    def __init__(self, folder, train, transform):
        self.folder = folder
        if train:
            self.path = os.path.join(folder, 'train')
        else:
            self.path = os.path.join(folder, 'val')
        self.imgs = traverse_dir(self.path)
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        # print('img path {}'.format(img))
        label = torch.tensor(0,dtype=torch.float) if 'ant' in img else torch.tensor(1,dtype=torch.float)
        # 图片尺寸大小不一致，需要利用transforms处理
        img = cv2.imread(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# train_dataset = MyDataset('/Users/wengyang/datasets_utils/hymenoptera_data', train=True)
# val_dataset = MyDataset('/Users/wengyang/datasets_utils/hymenoptera_data', train=False)
# train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
# test_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
#
# for imgs, labels in tqdm(train_loader):
#     pass
# for imgs, labels in tqdm(test_loader):
#     pass
