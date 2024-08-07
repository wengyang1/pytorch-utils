import os.path

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def traverse_dir(directory):
    items = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            items.append(os.path.join(root, name))
    return items


class MyDataset(Dataset):
    def __init__(self, folder, train):
        self.folder = folder
        if train:
            self.path = os.path.join(folder, 'train')
        else:
            self.path = os.path.join(folder, 'val')
        self.imgs = traverse_dir(self.path)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = 0 if 'ant' in img else 1
        return img, label

    def __len__(self):
        return len(self.imgs)


train_dataset = MyDataset('/Users/wengyang/datasets/hymenoptera_data', train=True)
val_dataset = MyDataset('/Users/wengyang/datasets/hymenoptera_data', train=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)

for imgs, labels in tqdm(train_loader):
    pass
for imgs, labels in tqdm(test_loader):
    pass
