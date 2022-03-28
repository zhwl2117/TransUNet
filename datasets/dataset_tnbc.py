import os
import random
from typing import Iterator
from PIL import Image
import torch
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def to_2digits(num: int):
    num = str(num)
    if len(num < 2):
        return '0' + num
    else:
        return num

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def b_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('1')

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = np.clip(image, -125, 275)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
        # label = (image - np.min(label)) / (np.max(label) - np.min(label) + 1e-6)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class DataGenerator(object):
    def __init__(self, dataset, batch_size=64) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        idxs = np.random.choice(len(self.dataset), self.batch_size, replace=True)
        images, labels = [], []
        for idx in idxs:
            sample = self.dataset[idx]
            images.append(sample['image'])
            labels.append(sample['label'])
        images = torch.stack(images)
        labels = torch.stack(labels)
        return {'image': images, 'label': labels}

    def __len__(self):
        return len(self.dataset)

class TNBC_dataset(Dataset):
    def __init__(self, base_dir, split=0.8, input_size=512, train=True, mode='L', transform=None):
        # super.__init__(TNBC_dataset)
        self.transform = transform  # using transform in torch!
        self.data_dir = base_dir
        self.sample_list = {}
        for path, _, files in os.walk(self.data_dir):
            for file in files:
                if file in self.sample_list.keys():
                    if 'image' in self.sample_list[file].keys():
                        self.sample_list[file]['label'] = os.path.join(path, file)
                    else:
                        self.sample_list[file]['image'] = os.path.join(path, file)
                else:
                    if 'Slide' in path:
                        self.sample_list[file] = {'image': os.path.join(path, file)}
                    else:
                        self.sample_list[file] = {'label': os.path.join(path, file)}
        self.sample_list = list(self.sample_list.values())
        split = int(len(self.sample_list) * split)
        random.shuffle(self.sample_list)
        self.train_list = self.sample_list[:split]
        self.test_list = self.sample_list[split:]
        self.train = train
        if transform is not None:
            self.transform = transform
        else:
            if self.train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop((input_size), scale=(0.2, 1.0), interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation([-20, 20]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.Resize((input_size, input_size)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        if mode == 'RGB':
            self.loader = rgb_loader
        else:
            self.loader = l_loader
        self.label_loader = b_loader

    def __len__(self):
        return len(self.train_list) if self.train else len(self.test_list)

    def __getitem__(self, idx):
        sample_list = self.train_list if self.train else self.test_list
        image_path, label_path = sample_list[idx]['image'], sample_list[idx]['label']
        image, label = self.loader(image_path), self.label_loader(label_path)
        image, label = np.array(image).astype(np.float32), np.array(label).astype(np.float32)
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    db_train = TNBC_dataset(base_dir=r'./data/tnbc', transform=transforms.Compose(
                                [RandomGenerator(output_size=[512, 512])]))
    print("The length of train set is: {}".format(len(db_train)))
    # train_loader = DataLoader(db_train, 64, shuffle=True)
    train_loader = DataGenerator(db_train, 64)
    print(len(train_loader))
    for _ in range(1):
        # print(len(sample))
        sample = next(train_loader)
        image, label = sample['image'], sample['label']
