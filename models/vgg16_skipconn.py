import torch
import torch.nn as nn
import torch.nn.functional as F

def block(in_,out_,blk,kernel_size=3,stride=1,padding=1,pool=True):
    layers = []
    for i in range(blk):
        conv2d = nn.Conv2d(in_, out_, kernel_size=3, padding=1)
        #layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [conv2d, nn.BatchNorm2d(out_), nn.ReLU(inplace=True)]
        in_ = out_
    if(pool):
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)

class vgg16_skip(nn.Module):

    def __init__(self, num_classes=10, init_weights=True):
        super(vgg16_skip, self).__init__()
        self.conv1 = block(3,64,2)
        self.conv2 = block(64,128,2,pool=False)
        self.conv3 = block(128,256,3,pool=False)
        self.conv4 = block(256,512,3,pool=False)
        self.conv5 = block(512,512,3)
        self.downsample = nn.Sequential(nn.Conv2d(64, 512, kernel_size=1, stride=2, bias=False))
        #self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


    def forward(self,x):

        identity = self.conv1(x)
        y = self.conv2(identity)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        #y = self.avgpool(y)

        # B,64,28,28 | B,3,3
        #print(y.shape, identity.shape)
        identity = self.downsample(identity)
        #print(y.shape, identity.shape)
        y = F.relu(y + identity)

        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return F.softmax(y,dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class vgg16_skip2(nn.Module):

    def __init__(self, num_classes=10, init_weights=True):
        super(vgg16_skip2, self).__init__()
        self.conv1 = block(3,64,2)
        self.conv2 = block(64,128,2,pool=False)
        self.conv3 = block(128,256,3,pool=False)
        self.conv4 = block(256,512,3,pool=False)
        self.conv5 = block(512,512,3,pool=False)
        self.downsample = nn.Sequential(nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(512))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


    def forward(self,x):

        identity = self.conv1(x)
        y = self.conv2(identity)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        #y = self.avgpool(y)

        # B,64,28,28 | B,3,3
        #print(y.shape, identity.shape)
        identity = self.downsample(identity)
        #print(y.shape, identity.shape)
        y = F.relu(y + identity)
        y = self.avgpool(y)

        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return F.softmax(y,dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__=="__main__":
    #from torchviz import make_dot
    model = vgg16_skip2()
    x = torch.zeros(4, 3, 28, 28, dtype=torch.float, requires_grad=False)
    out = model(x)
    print("out shape : ",out.shape)
