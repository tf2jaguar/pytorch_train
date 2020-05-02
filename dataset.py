import os

import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__image_net_stats = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}


class CarDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        CarDataset: Custom dataset
        :param csv_file（字符串）:带有注释的csv文件的路径。
        :param root_dir（字符串）:包含所有图像的根目录。
        :param transform（可调用,可选）:应用于样本的可选transform。
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[index, 0])
        label = self.landmarks_frame.iloc[index, 1:]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, int(label)

    def __len__(self):
        return len(self.landmarks_frame)


def inception_preproccess(input_size, normalize=None):
    if normalize is None:
        normalize = __image_net_stats
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None, normalize=None):
    if normalize is None:
        normalize = __image_net_stats
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform(augment=True, input_size=224):
    normalize = __image_net_stats
    scale_size = int(input_size / 0.875)
    if augment:
        return inception_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


def get_loaders(dataroot, val_batch_size, train_batch_size, input_size, workers):
    val_data = CarDataset(dataroot + '/val.txt', './data', transform=get_transform(True, input_size))
    # val_data = datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=get_transform(False, input_size))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)

    # train_data = datasets.ImageFolder(root=os.path.join(dataroot, 'train'),
    #                                   transform=get_transform(input_size=input_size))

    train_data = CarDataset(dataroot + '/train.txt', './data', transform=get_transform(True, input_size))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    return train_loader, val_loader


def get_test_loaders(dataroot, batch_size, input_size, workers):
    test_data = CarDataset(dataroot + '/test.txt', './data', transform=get_transform(True, input_size))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True)
    return test_loader
