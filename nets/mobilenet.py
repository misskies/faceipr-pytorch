from nets.model import EqualLinear, Blur
import torch
import torch.nn as nn
import math
import sys

sys.path.append("..")
from op import conv2d_gradfix


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        group=0,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            # self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(512, in_channel, bias_init=1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )
#11
    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        if style != None:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            weight = self.scale * self.weight * style
        else:
            style=torch.ones(batch,1,in_channel,1,1)
            style=style.to(self.weight.device)
            weight = self.weight*style

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.downsample:
            #input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=1, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding,stride=1, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out





def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )


class watermark_MobileNetV1(nn.Module):
    def __init__(self,watermark_size=32):
        super(watermark_MobileNetV1, self).__init__()
        self.Md1=ModulatedConv2d(3,32,3,watermark_size,downsample=True)
        self.BN1=nn.BatchNorm2d(32)
        #1

        self.Md2=ModulatedConv2d(32,32,3,watermark_size)
        self.BN2=nn.BatchNorm2d(32)
        self.Md2_1=ModulatedConv2d(32,64,1,watermark_size)
        self.BN2_1=nn.BatchNorm2d(64)
        #2

        self.Md3=ModulatedConv2d(64,64,3,watermark_size,downsample=True)
        self.BN3=nn.BatchNorm2d(64)
        self.Md3_1=ModulatedConv2d(64,128,1,watermark_size)
        self.BN3_1=nn.BatchNorm2d(128)
        #3

        self.Md4=ModulatedConv2d(128,128,3,watermark_size)
        self.BN4=nn.BatchNorm2d(128)
        self.Md4_1=ModulatedConv2d(128,128,1,watermark_size)
        self.BN4_1=nn.BatchNorm2d(128)
        #4

        self.Md5=ModulatedConv2d(128,128,3,watermark_size,downsample=True)
        self.BN5=nn.BatchNorm2d(128)
        self.Md5_1=ModulatedConv2d(128,256,1,watermark_size)
        self.BN5_1=nn.BatchNorm2d(256)
        #5

        self.Md6=ModulatedConv2d(256,256,3,watermark_size)
        self.BN6=nn.BatchNorm2d(256)
        self.Md6_1=ModulatedConv2d(256,256,1,watermark_size)
        self.BN6_1=nn.BatchNorm2d(256)
        #6
        #stage1

        self.Md7=ModulatedConv2d(256,256,3,watermark_size,downsample=True)
        self.BN7=nn.BatchNorm2d(256)
        self.Md7_1=ModulatedConv2d(256,512,1,watermark_size)
        self.BN7_1=nn.BatchNorm2d(512)
        #7

        self.Md8=ModulatedConv2d(512,512,3,watermark_size)
        self.BN8=nn.BatchNorm2d(512)
        self.Md8_1=ModulatedConv2d(512,512,1,watermark_size)
        self.BN8_1=nn.BatchNorm2d(512)

        self.Md9=ModulatedConv2d(512,512,3,watermark_size)
        self.BN9=nn.BatchNorm2d(512)
        self.Md9_1=ModulatedConv2d(512,512,1,watermark_size)
        self.BN9_1=nn.BatchNorm2d(512)

        self.Md10=ModulatedConv2d(512,512,3,watermark_size)
        self.BN10=nn.BatchNorm2d(512)
        self.Md10_1=ModulatedConv2d(512,512,1,watermark_size)
        self.BN10_1=nn.BatchNorm2d(512)

        self.Md11=ModulatedConv2d(512,512,3,watermark_size)
        self.BN11=nn.BatchNorm2d(512)
        self.Md11_1=ModulatedConv2d(512,512,1,watermark_size)
        self.BN11_1=nn.BatchNorm2d(512)

        self.Md12=ModulatedConv2d(512,512,3,watermark_size)
        self.BN12=nn.BatchNorm2d(512)
        self.Md12_1=ModulatedConv2d(512,512,1,watermark_size)
        self.BN12_1=nn.BatchNorm2d(512)
        #8-12

        self.Md13=ModulatedConv2d(512,512,3,watermark_size,downsample=True)
        self.BN13=nn.BatchNorm2d(512)
        self.Md13_1=ModulatedConv2d(512,1024,1,watermark_size)
        self.BN13_1=nn.BatchNorm2d(1024)
        #13

        self.Md14=ModulatedConv2d(1024,1024,3,watermark_size)
        self.BN14=nn.BatchNorm2d(1024)
        self.Md14_1=ModulatedConv2d(1024,1024,1,watermark_size)
        self.BN14_1=nn.BatchNorm2d(1024)
        #14
        self.relu=nn.ReLU6()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,watermark_in):
        x=self.Md1(x,watermark_in)
        x=self.BN1(x)
        x=self.relu(x)
        #1

        x=self.Md2(x,watermark_in)
        x=self.BN2(x)
        x=self.relu(x)
        x=self.Md2_1(x,watermark_in)
        x=self.BN2_1(x)
        x=self.relu(x)
        #2

        x=self.Md3(x,watermark_in)
        x=self.BN3(x)
        x=self.relu(x)
        x=self.Md3_1(x,watermark_in)
        x=self.BN3_1(x)
        x=self.relu(x)
        #3

        x=self.Md4(x,watermark_in)
        x=self.BN4(x)
        x=self.relu(x)
        x=self.Md4_1(x,watermark_in)
        x=self.BN4_1(x)
        x=self.relu(x)
        #4

        x=self.Md5(x,watermark_in)
        x=self.BN5(x)
        x=self.relu(x)
        x=self.Md5_1(x,watermark_in)
        x=self.BN5_1(x)
        x=self.relu(x)
        #5

        x=self.Md6(x,watermark_in)
        x=self.BN6(x)
        x=self.relu(x)
        x=self.Md6_1(x,watermark_in)
        x=self.BN6_1(x)
        x=self.relu(x)
        #6
        #stage1

        x=self.Md7(x,watermark_in)
        x=self.BN7(x)
        x=self.relu(x)
        x=self.Md7_1(x,watermark_in)
        x=self.BN7_1(x)
        x=self.relu(x)
        #7
        x = self.Md8(x, watermark_in)
        x = self.BN8(x)
        x = self.relu(x)
        x = self.Md8_1(x, watermark_in)
        x = self.BN8_1(x)
        x = self.relu(x)

        x = self.Md9(x, watermark_in)
        x = self.BN9(x)
        x = self.relu(x)
        x = self.Md9_1(x, watermark_in)
        x = self.BN9_1(x)
        x = self.relu(x)

        x = self.Md10(x, watermark_in)
        x = self.BN10(x)
        x = self.relu(x)
        x = self.Md10_1(x, watermark_in)
        x = self.BN10_1(x)
        x = self.relu(x)

        x = self.Md11(x, watermark_in)
        x = self.BN11(x)
        x = self.relu(x)
        x = self.Md11_1(x, watermark_in)
        x = self.BN11_1(x)
        x = self.relu(x)

        x = self.Md12(x, watermark_in)
        x = self.BN12(x)
        x = self.relu(x)
        x = self.Md12_1(x, watermark_in)
        x = self.BN12_1(x)
        x = self.relu(x)
        #8-12

        # x=self.stage2(x)
        x=self.Md13(x,watermark_in)
        x=self.BN13(x)
        x=self.relu(x)
        x=self.Md13_1(x,watermark_in)
        x=self.BN13_1(x)
        x=self.relu(x)
        #13

        x=self.Md14(x,watermark_in)
        x=self.BN14(x)
        x=self.relu(x)
        x=self.Md14_1(x,watermark_in)
        x=self.BN14_1(x)
        x=self.relu(x)
        #14
        # x=self.stage3(x)
        return x


        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.avg(x)
        # # x = self.model(x)
        # x = x.view(-1, 1024)
        # x = self.fc(x)
        # return x


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2), #dowmsample
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1), 

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # # x = self.model(x)
        # x = x.view(-1, 1024)
        # x = self.fc(x)
        return x


