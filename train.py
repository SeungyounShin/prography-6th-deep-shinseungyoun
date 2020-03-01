import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.vgg16_skipconn import *
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

batch_size = 64
epoch = 15

# Normalize data with mean=0.5, std=1.0
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.Normalize((0.5,), (1.0,))
])

download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True) #60000
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True) #10000

train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                         batch_size=batch_size,
                         shuffle=True)

model = vgg16_skip2(10)
model = model.cuda()
# loss
criterion = nn.CrossEntropyLoss()
# backpropagation method
learning_rate = 1e-5
opt = optim.Adam(model.parameters(), lr=learning_rate)

losses = list()
for e in range(epoch):
    for batch_idx, (x, target) in enumerate(train_loader):
        opt.zero_grad()

        x,target = x.cuda(),target.cuda()
        out = model(x)
        loss = criterion(out, target)

        loss.backward()
        opt.step()

        if (batch_idx+1) % 50 == 0:
            losses.append(loss.item())

            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} ".format(
                e+1, epoch, batch_idx+1, 60000//batch_size, loss.item() ))
    torch.save(model, "./ckpt/"+str(e)+"_ver2.pth")
