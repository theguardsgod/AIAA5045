import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import dataset
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)


re_size = 256
input_size = 224
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]



def get_fixMatch(configs_dict):
    transform_labeled = transforms.Compose([
        transforms.Resize(re_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3]),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = dataset.Skin7(root="/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Training_Input", train='train',
                             transform=train_transform)
    valset = dataset.Skin7(root="/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Validation_Input", train='val',
                           transform=val_transform)
    testset = dataset.Skin7(root="/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Test_Input", train='test',
                            transform=val_transform)
    root = "/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Validation_Input"
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        configs_dict, trainset.targets)

    train_labeled_dataset = Skin7SSL(
        root, train_labeled_idxs, train='train',
        transform=transform_labeled)

    train_unlabeled_dataset = Skin7SSL(
        root, train_unlabeled_idxs, train='train',
        transform=TransformFixMatch(mean=mean, std=std))


    return train_labeled_dataset, train_unlabeled_dataset, valset, testset




def x_u_split(configs_dict, labels):
    label_per_class = configs_dict['num_labeled'] // configs_dict['num_classes']
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(configs_dict['num_classes']):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == configs_dict['num_labeled']

    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size)])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class Skin7SSL(dataset.Skin7):
    def __init__(self, root, indexs, train,
                 transform=None):
        super().__init__(root, train=train,
                         transform=transform,
                         )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




