import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.vgg16_skipconn import vgg16_skip
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

batch_size  = 1
model_path = "./ckpt/3_ver2.pth"

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.Normalize((0.5,), (1.0,))
])

download_root = './MNIST_DATASET'

test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

testlen = len(test_dataset)

model = torch.load(model_path,map_location='cpu')
model = model.eval()


for batch_idx, (x, target) in enumerate(test_loader):
    out = model(x)
    print(out)
    print(target)
    break

"""
right = 0
for batch_idx, (x, target) in enumerate(test_loader):
    out = model(x)
    r = (out.max(dim=1)[1] == target).sum()
    print(r/float(batch_size))
    right += r

print("acc : ",right/float(testlen))
"""
