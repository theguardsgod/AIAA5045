"""Trainer

    Train all your model here.
"""

import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.function import init_logging, init_environment, get_lr, \
    print_loss_sometime
from utils.metric import mean_class_recall
import config
import dataset
import model
from loss import class_balanced_loss
from rich import print
from FixMatchDataset import get_fixMatch
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
mu = configs_dict["mu"]
threshold = configs_dict["threshold"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_environment(seed=seed, cuda_id=cuda_ids)
_print = init_logging(log_dir, exp).info
configs.print_config(_print)
tf_log = os.path.join(log_dir, exp)
writer = SummaryWriter(log_dir=tf_log)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# Pre-peocessed input image
if backbone in ["resnet50", "resnet18"]:
    re_size = 300
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
elif backbone in ["dense121","dense201"]:
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


# modify here to change pretrain
net = model.Network(backbone=backbone, num_classes=num_classes,
                    input_channel=input_channel, pretrained=True)

_print("=> Using device ids: {}".format(cuda_ids))
device_ids = list(range(len(cuda_ids.split(","))))

labeled_dataset, unlabeled_dataset, valset, testset = get_fixMatch(configs_dict)

train_sampler = val_sampler = None
train_sampler = RandomSampler
if len(device_ids) == 1:
    _print("Model single cuda")
    net = net.to(device)
else:
    _print("Model parallel !!")
    # torch.distributed.init_process_group(backend="nccl")
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    # net = torch.nn.parallel.DistributedDataParallel(net)
    net = nn.DataParallel(net, device_ids=device_ids).to(device)

_print("=> iter_fold is {}".format(iter_fold))
trainset = dataset.Skin7(root="/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Training_Input", train='train',
                         transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers,
                                          sampler=None)
labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
         pin_memory=True,
        drop_last=True)

unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=batch_size*mu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers,
                                        sampler=val_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers,
                                        sampler=val_sampler)

# Loss
if loss_fn == "WCE":
    _print("Loss function is WCE")
    weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
elif loss_fn == "CE":
    _print("Loss function is CE")
    criterion = nn.CrossEntropyLoss().to(device)
else:
    _print("Need loss function.")

# Optmizer
scheduler = None
if optimizer == "SGD":
    _print("=> Using optimizer SGD with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.1, patience=50, verbose=True,
                threshold=1e-4)
elif optimizer == "Adam":
    _print("=> Using optimizer Adam with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
else:
    _print("Need optimizer")
    sys.exit(-1)


start_epoch = 0
if resume:
    _print("=> Resume from model at epoch {}".format(resume))
    resume_path = os.path.join(model_dir, str(exp), str(resume))
    ckpt = torch.load(resume_path)
    net.load_state_dict(ckpt)
    start_epoch = resume + 1
else:
    _print("Train from scrach!!")


desc = "Exp-{}-Train".format(exp)
sota = {}
sota["epoch"] = start_epoch
sota["mcr"] = -1.0


labeled_iter = iter(labeled_trainloader)
unlabeled_iter = iter(unlabeled_trainloader)
eval_step = 10000 // (batch_size * mu)
weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
class_weights = torch.FloatTensor(weights).cuda()
for epoch in range(start_epoch+1, n_epochs+1):
    net.train()
    losses = []
    losses_x = []
    losses_u = []
    mask_probs = []

    for batch_idx in range(eval_step):


        try:
            inputs_x, targets_x = next(labeled_iter)
            # error occurs ↓
            # inputs_x, targets_x = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_iter)
            # error occurs ↓
            # inputs_x, targets_x = next(labeled_iter)

        try:
            (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            # error occurs ↓
            # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            # error occurs ↓
            # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
        batch_size = inputs_x.shape[0]
        inputs = interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*mu+1).to(device)
        targets_x = targets_x.to(device)
        logits = net(inputs)
        logits = de_interleave(logits, 2*mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits
        opt.zero_grad()
        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()

        loss = Lx + 0.5 * Lu
        loss.backward()
        opt.step()

    # print to log
    dicts = {
        "epoch": epoch, "n_epochs": n_epochs, "loss": loss.item()
    }
    print_loss_sometime(dicts, _print=_print)

    train_avg_loss = np.mean(losses)
    if scheduler is not None:
        scheduler.step(train_avg_loss)

    writer.add_scalar("Lr", get_lr(opt), epoch)
    writer.add_scalar("Loss/train/", train_avg_loss, epoch)

    if epoch % eval_frequency == 0:
        net.eval()
        y_true = []
        y_pred = []
        for _, (data, target) in enumerate(trainloader):
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)

        acc = accuracy_score(y_true, y_pred)
        mcr = mean_class_recall(y_true, y_pred)
        _print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
        _print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
        writer.add_scalar("Acc/train/", acc, epoch)
        writer.add_scalar("Mcr/train/", mcr, epoch)

        y_true = []
        y_pred = []
        for _, (data, target) in enumerate(valloader):
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)

        acc = accuracy_score(y_true, y_pred)
        mcr = mean_class_recall(y_true, y_pred)
        _print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
        _print("=> Epoch:{} - val mcr: {:.4f}".format(epoch, mcr))
        writer.add_scalar("Acc/val/", acc, epoch)
        writer.add_scalar("Mcr/val/", mcr, epoch)

        # Val acc
        if mcr > sota["mcr"]:
            sota["mcr"] = mcr
            sota["epoch"] = epoch
            model_path = os.path.join(model_dir, str(exp), str(epoch))
            best_model_path = os.path.join(model_dir, str(exp), "best")
            _print("=> Save model in {}".format(model_path))
            _print("=> Also save model in {}".format(best_model_path))
            net_state_dict = net.state_dict()
            torch.save(net_state_dict, model_path)
            torch.save(net_state_dict, best_model_path)
    if epoch == n_epochs:
        _print("=> Loading best model to test")
        best_ckp = torch.load(best_model_path)
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

        acc = accuracy_score(y_true, y_pred)
        mcr = mean_class_recall(y_true, y_pred)
        _print("=> Epoch:{} - test acc: {:.4f}".format(epoch, acc))
        _print("=> Epoch:{} - test mcr: {:.4f}".format(epoch, mcr))
        writer.add_scalar("Acc/test/", acc, epoch)
        writer.add_scalar("Mcr/test/", mcr, epoch)
        

_print("=> Finish Training")
_print("=> Best epoch {} with {} on Val: {:.4f}".format(sota["epoch"],
                                                        "sota",
                                                        sota["mcr"]))
