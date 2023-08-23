
import torch.nn as nn
import torchvision.models as models
from .md_resnet import ModulatedConv2d, mdresnet34

class ModifiedMdResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModifiedMdResNet34, self).__init__()
        resnet34 = mdresnet34()
        
        # 获取resnet34的所有层，直到倒数第二个残差块
        layers = list(resnet34.children())[:-2]
        self.head = layers[:4]
        self.features = nn.Sequential(*layers[4:])
        
        # 创建一个新的残差块，其输出通道数为1024
        self.new_mdconv_0 = ModulatedConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.new_bn_1 = nn.BatchNorm2d(512)
        self.new_relu_2 = nn.ReLU(inplace=True)
        self.new_mdconv_3 = ModulatedConv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.new_bn_4 = nn.BatchNorm2d(1024)

        # 残差连接，调整输入的维度以匹配new_layer的输出
        self.shortcut = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        
        # 全连接层
        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, watermark):
        
        x = self.head[0](x, watermark) # first Mdconv2d
        x = self.head[1](x)
        x = self.head[2](x)
        x = self.head[3](x)
        
        x = self.features((x, watermark)) # return feature_map and watermark
        
        # 残差连接
        residual = self.shortcut(x[0])
        # print(x[0]) 
        x = self.new_mdconv_0(x[0], watermark)
        x = self.new_bn_1(x)
        x = self.new_mdconv_3(x, watermark)
        x = self.new_bn_4(x)

        
        x += residual
        x = nn.ReLU(inplace=True)(x)
        
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
# 