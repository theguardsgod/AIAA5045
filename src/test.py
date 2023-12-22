"""Trainer

    Train all your model here.
"""

import torch
import os
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score


from utils.function import init_logging, init_environment, get_lr, \
    print_loss_sometime
from utils.metric import mean_class_recall, eval_matrix
import config
import dataset
import model
from loss import class_balanced_loss
from rich import print

configs = config.Config()
configs_dict = configs.get_config()
# Load hyper parameter from config file
exp = configs_dict["experiment_index"]
cuda_ids = configs_dict["cudas"]
num_workers = configs_dict["num_workers"]
seed = configs_dict["seed"]
n_epochs = configs_dict["n_epochs"]
log_dir = configs_dict["log_dir"]
model_dir = configs_dict["model_dir"]
batch_size = configs_dict["batch_size"]
learning_rate = configs_dict["learning_rate"]
backbone = configs_dict["backbone"]
eval_frequency = configs_dict["eval_frequency"]
resume = configs_dict["resume"]
optimizer = configs_dict["optimizer"]
initialization = configs_dict["initialization"]
num_classes = configs_dict["num_classes"]
iter_fold = configs_dict["iter_fold"]
loss_fn = configs_dict["loss_fn"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_environment(seed=seed, cuda_id=cuda_ids)
_print = init_logging(log_dir, exp).info
configs.print_config(_print)
tf_log = os.path.join(log_dir, exp)
writer = SummaryWriter(log_dir=tf_log)


# Pre-peocessed input image
if backbone in ["resnet50", "resnet18"]:
    re_size = 300
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
elif backbone in ["dense121"]:
    re_size = 256
    input_size = 224
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
elif backbone in ["NASNetALarge", "PNASNet5Large"]:
    re_size = 441
    input_size = 331
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
else:
    _print("Need backbone")
    sys.exit(-1)

_print("=> Image resize to {} and crop to {}".format(re_size, input_size))

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
testset = dataset.Skin7(root="/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Test_Input", train='test',
                       transform=val_transform)

net = model.Network(backbone=backbone, num_classes=num_classes,
                    input_channel=input_channel, pretrained=True)

_print("=> Using device ids: {}".format(cuda_ids))
device_ids = list(range(len(cuda_ids.split(","))))
train_sampler = val_sampler = None
if len(device_ids) == 1:
    _print("Model single cuda")
    net = net.to(device)
else:
    _print("Model parallel !!")
    net = nn.DataParallel(net, device_ids=device_ids).to(device)


testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers,
                                        sampler=val_sampler)


desc = "Exp-{}-TEST".format(exp)


_print("=> Loading best model to test")
best_ckp = torch.load('/home/ubuntu22/code/AIAA5045/saved/models/007/best')
net.load_state_dict(best_ckp)
net.eval()
y_true = []
y_pred = []
for _, (data, target) in enumerate(testloader):
    data = data.to(device)
    predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
    y_pred.extend(predict)
    target = target.cpu().data.numpy()
    y_true.extend(target)
# print(y_true[0])
# print(y_pred[0])
# import ipdb; ipdb.set_trace()
Accus, Pre, Senss, Specs, F1, AUROCs = eval_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
mcr = mean_class_recall(y_true, y_pred)
epoch = 0
_print("=> Epoch:{} - test acc: {:.4f}".format(epoch, acc))
_print("=> Epoch:{} - test mcr: {:.4f}".format(epoch, mcr))
_print("=> Epoch:{} - test AUROC: {:.4f}".format(epoch, AUROCs))
_print("=> Epoch:{} - test Accu: {:.4f}".format(epoch, Accus))
_print("=> Epoch:{} - test Sens: {:.4f}".format(epoch, Senss))
_print("=> Epoch:{} - test Spec: {:.4f}".format(epoch, Specs))
_print("=> Epoch:{} - test Pre: {:.4f}".format(epoch, Pre))
_print("=> Epoch:{} - test F1: {:.4f}".format(epoch, F1))

writer.add_scalar("Acc/test/", acc, epoch)
writer.add_scalar("Mcr/test/", mcr, epoch)
writer.add_scalar("AUROC/test/", AUROCs, epoch)
writer.add_scalar("Accu/test/", Accus, epoch)
writer.add_scalar("Sens/test/", Senss, epoch)
writer.add_scalar("Spec/test/", Specs, epoch)
writer.add_scalar("Pre/test/", Pre, epoch)
writer.add_scalar("F1/test/", F1, epoch)       

_print("=> Finish Testing")