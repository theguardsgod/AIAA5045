import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import random
import numpy as np
import dataset
import copy
from tqdm import tqdm

##  设置参数  ##
learning_rate = 0.01

# 设置种子
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 如果您使用CUDA，则还需要设置以下内容
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 数据预处理,数据增强 ##
re_size = 300
input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
results_file = open("training_resnet_without.txt", "w")


train_transform = transforms.Compose([
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
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

input_channel = 3
trainset = dataset.Skin7(root="ISIC2018/ISIC2018_Task3_Training_Input", train='train',
                         transform=train_transform)
valset = dataset.Skin7(root="ISIC2018/ISIC2018_Task3_Validation_Input", train='val',
                       transform=val_transform)
testset = dataset.Skin7(root="ISIC2018/ISIC2018_Task3_Test_Input", train='test',
                       transform=val_transform)
## 创建模型 ##
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7) # 修改最后的全连接层

# 定义损失函数和优化器
weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

batch_size = 32
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 将模型移至设备
model = model.to(device)

# 准确率计算函数
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels.data).double() / len(labels)

def train_model(model, criterion, optimizer, num_epochs=25, file_path='training_log.txt'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

            # 每个epoch有两个阶段：train 和 val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            running_loss = 0.0
            running_corrects = 0

            print(f"Epoch {epoch+1}/{num_epochs}")

            # 迭代数据
            for inputs, labels in tqdm((train_loader if phase == 'train' else val_loader), desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 反向传播 + 优化仅在训练阶段
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            results = 'Epoch [{}/{}], Loss: {:.4f}, Accuracy: {epoch_acc:.4f}'
            results_file.write(results)


            # 深拷贝并保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model_res_without_100.pth')

        print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 开始训练模型
trained_model = train_model(model, criterion, optimizer, num_epochs=100, file_path='training_log.txt')

print('Finished Training') 
results_file.close()