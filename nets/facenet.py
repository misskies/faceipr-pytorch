
import math

import torch
import torch.nn as nn
from sympy.printing.tests.test_tensorflow import tf
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1, watermark_MobileNetV1
from nets.baseline import post_embed_watermark ,post_extract_watermark
import nets.model as md
from nets.resnet import ModifiedMdResNet34

def linear(inp,oup):
    return nn.Sequential(
        nn.Linear(inp,oup),
        nn.BatchNorm1d(oup),
        nn.ReLU()
    )
class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)


    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x

class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x

class E_watermark(nn.Module):
    def __init__(self,watermark_size=32, n_mlp=8, lr_mlp=0.01):
        super(E_watermark, self).__init__()
        layers = []
        style_dim=512
        layers.append(
            md.EqualLinear(
                watermark_size, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
            )
        )
        for i in range(n_mlp-1):
            layers.append(
                md.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        self.style = nn.Sequential(*layers)
    def forward(self,watermark_in):

        watermark_out = self.style(watermark_in)
        return watermark_out
def Noise_injection(feature_in,robustness="none",noise_power=0.1):
    noise_scale = noise_power
    flip_prob = noise_power
    round_scale = int(noise_power)
    if robustness == "noise":
        # the noise scale is set to 0.1 temporarily
        feature_perb = feature_in + torch.randn_like(feature_in) * noise_scale
    elif robustness == "flip":
        # each sign of feature_in has 10% probability to be flipped
        mask = torch.rand_like(feature_in) < flip_prob
        feature_perb = feature_in * (-1) ** mask
    elif robustness == "combine":
        mask = torch.rand_like(feature_in) < flip_prob
        feature_perb = feature_in * (-1) ** mask + torch.randn_like(feature_in) * noise_scale
    elif robustness == "round" :
        assert round_scale >= 0
        feature_in = feature_in * (10**round_scale)
        feature_in = torch.round(feature_in)
        feature_perb = feature_in / (10**round_scale)
    elif robustness == "random_del" :
        random = torch.randn_like(feature_in)
        zero = torch.zeros_like(random)
        one = torch.ones_like(random)
        random = torch.where(random <= noise_power,zero,random)
        random = torch.where(random > noise_power, one, random)
        feature_perb = random * feature_in
    elif robustness == "none":
        feature_perb = feature_in
    else:
        raise ValueError("robustness should be one of [noise, flip, combine,round,random_del,none]")
    return  feature_perb


class D_watermark(nn.Module):
    def __init__(self,embedding_size=1024,watermark_size=32,lr_mlp=0.01, robustness="none"):
        super(D_watermark,self).__init__()
        self.stage=nn.Sequential(
            linear(embedding_size,512),
            linear(512,256),
            linear(256,128),
            linear(128,64),
            nn.Linear(64,watermark_size)
        )
        self.robustness = robustness
        # layers =[]
        # in_dim =1024
        # out_dim=512
        # for i in range(3):
        #     layers.append(
        #         md.EqualLinear(
        #             in_dim, out_dim, lr_mul=lr_mlp, activation="fused_lrelu"
        #         )
        #     )
        #     in_dim =int(out_dim)
        #     out_dim=int(out_dim/2)
        # layers.append(
        #     nn.Linear(in_dim,watermark_size)
        # )
        # self.style=nn.Sequential(*layers)
    def forward(self,feature_in):

        watermark_out=self.stage(feature_in)
        #watermark_out=self.style(feature_in)
        return watermark_out
    
    
class D_watermark_128(nn.Module):
    def __init__(self,watermark_size=32,lr_mlp=0.01, robustness="none"):
        super(D_watermark_128,self).__init__()
        self.stage=nn.Sequential(
            linear(128,128),
            linear(128,64),
            linear(64,32),
            linear(32,16),
            nn.Linear(16,watermark_size)
        )
        self.robustness = robustness
    def forward(self,feature_in):

        watermark_out=self.stage(feature_in)
        #watermark_out=self.style(feature_in)
        return watermark_out
    


class D_watermark_conv(nn.Module):
    def __init__(self, watermark_size=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, watermark_size)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add an extra dimension for the channels
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


class D_watermark_attn(nn.Module):
    def __init__(self, embedding_size=1024, watermark_size=32):
        super().__init__()
        self.query = nn.Linear(embedding_size, watermark_size)
        self.key = nn.Linear(embedding_size, watermark_size)
        self.value = nn.Linear(embedding_size, watermark_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5), dim=-1)
        return attention_weights @ V




class D_watermark_convattn(nn.Module):
    def __init__(self, watermark_size=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.attention = SelfAttention(64 * 256, 128)
        self.fc = nn.Linear(128, watermark_size)

    def forward(self, x):
        x = x.unsqueeze(1) # Add an extra dimension for the channels
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.attention(x)
        x = self.fc(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        Q = self.query(x)
        Q = Q.unsqueeze(0)
        K = self.key(x)
        K = K.unsqueeze(0)
        V = self.value(x)
        V = V.unsqueeze(0)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5), dim=-1)
        return (attention_weights @ V).squeeze(0)




class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [md.ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(md.ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = md.ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            md.EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            md.EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class Facenet(nn.Module):
    def __init__(self,backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False,watermark_size=32, robustness="none",noise_power=0.1, decoder_arch="fc"):
        super(Facenet, self).__init__()

        if backbone == "mobilenet":
            self.backbone =watermark_MobileNetV1(watermark_size=watermark_size)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        elif backbone == "resnet34":
            self.backbone =ModifiedMdResNet34()
            flat_shape = 1024
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.robustness=robustness
        self.noise_power=noise_power
        self.watermark_Encoder=E_watermark(watermark_size=watermark_size)
        
        if decoder_arch == "fc": 
            self.watermark_Decoder=D_watermark(embedding_size=flat_shape, watermark_size=watermark_size, robustness=robustness)
        elif decoder_arch == "conv":
            self.watermark_Decoder = D_watermark_conv(watermark_size=watermark_size)
        elif decoder_arch == "attn":
            self.watermark_Decoder = D_watermark_attn(embedding_size=flat_shape, watermark_size=watermark_size)
        elif decoder_arch == "conv_attn":
            self.watermark_Decoder = D_watermark_convattn(watermark_size=watermark_size)
        else:
            raise ValueError(f"Unsupported decoder architecture - `{decoder_arch}`, Use fc, conv, attn, conv_attn.")
        
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x,watermark_in,mode = "predict"):
        if mode == 'predict':
            watermark_out = self.watermark_Encoder(watermark_in)
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x=Noise_injection(x,robustness=self.robustness,noise_power=self.noise_power)
            watermark_fin=self.watermark_Decoder(x)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x,watermark_fin

        if mode == 'origin':
            watermark_out = None
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            before_normalize = self.last_bn(x)  # feature
            x = F.normalize(x, p=2, dim=1)
            cls = self.classifier(before_normalize)
            return x,cls

        if mode == 'Unmd_predict':
            watermark_out = None
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            watermark_fin=self.watermark_Decoder(x)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x,watermark_fin
        if mode in ['LSB', 'FFT', 'Noise'] :
            watermark_out=None
            watermark_size = watermark_in.shape[1]
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = post_embed_watermark(x,watermark_in, mode) #watermakred_embedding
            x=Noise_injection(x,robustness=self.robustness,noise_power=self.noise_power)
            watermark_fin= post_extract_watermark(x, watermark_size, mode=mode)

            if torch.is_complex(x):
                x = x.to(torch.float32)
            
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x,watermark_fin

        watermark_out=self.watermark_Encoder(watermark_in)
        x = self.backbone(x,watermark_out)
        #x(5,5,1024)
        x = self.avg(x)
        #x(1,1,1024)
        x = x.view(x.size(0), -1)#(batch,1024)
        watermark_fin = self.watermark_Decoder(x)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)#feature
        #add Decoder
        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls,watermark_fin

    def forward_feature(self, x,watermark_in):
        watermark_out = self.watermark_Encoder(watermark_in)
        x = self.backbone(x, watermark_out)
        #x(5,5,1024)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        watermark_fin = self.watermark_Decoder(x)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x,watermark_in

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x

class Facenet_128(nn.Module):
    def __init__(self,backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False,watermark_size=32, robustness="none",noise_power=0.1):
        super(Facenet_128, self).__init__()

        if backbone == "mobilenet":
            self.backbone =watermark_MobileNetV1(watermark_size=watermark_size)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.robustness=robustness
        self.noise_power=noise_power
        self.watermark_Encoder=E_watermark(watermark_size=watermark_size)
        self.watermark_Decoder=D_watermark_128(watermark_size=watermark_size, robustness=robustness)
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x,watermark_in,mode = "predict"):
        if mode == 'predict':
            watermark_out = self.watermark_Encoder(watermark_in)
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)

            x=Noise_injection(x,robustness=self.robustness,noise_power=self.noise_power)
            watermark_fin=self.watermark_Decoder(x)
            return x,watermark_fin

        if mode == 'origin':
            watermark_out = None
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            before_normalize = self.last_bn(x)  # feature
            x = F.normalize(x, p=2, dim=1)
            cls = self.classifier(before_normalize)
            return x,cls

        if mode == 'Unmd_predict':
            watermark_out = None
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)

            watermark_fin=self.watermark_Decoder(x)
            return x,watermark_fin

        if mode == 'LSB':
            watermark_out=None
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            
            x = embed_watermark(x,watermark_in) #watermakred_embedding
            x=Noise_injection(x,robustness=self.robustness,noise_power=self.noise_power)
            watermark_fin= extract_watermark(x,1024)

            return x,watermark_fin

        watermark_out=self.watermark_Encoder(watermark_in)
        x = self.backbone(x,watermark_out)
        #x(5,5,1024)
        x = self.avg(x)
        #x(1,1,1024)
        x = x.view(x.size(0), -1)#(batch,1024)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)#feature
        #add Decoder
        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        watermark_fin = self.watermark_Decoder(x)
        return x, cls,watermark_fin

    def forward_feature(self, x,watermark_in):
        watermark_out = self.watermark_Encoder(watermark_in)
        x = self.backbone(x, watermark_out)
        #x(5,5,1024)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        watermark_fin = self.watermark_Decoder(x)
        return before_normalize, x,watermark_in

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x






class Facenet_loss(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False):
        super(Facenet_loss, self).__init__()
        if backbone == "mobilenet":
            # self.backbone = mobilenet(pretrained)
            self.backbone = watermark_MobileNetV1(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)
        self.Tanh=nn.Tanh()
    def forward(self, x, watermark_in=None, mode = "predict"):
        if mode == 'predict':
            watermark_out = None
            x = self.backbone(x, watermark_out)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x, x

        watermark_out = None
        x = self.backbone(x, watermark_out)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

    def forward_feature(self, x):
        watermark_out = None
        x = self.backbone(x, watermark_out)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x


#if __name__ == '__main__':
    # a = torch.empty(3, 32).uniform_(0, 1)
    # a=torch.bernoulli(a)
    # x=torch.rand(3,3,160,160)
    # facenet=Facenet(backbone="mobilenet",num_classes=2,pretrained=False)
    # x,y=facenet(x,a)
    # m=torch.nn.Sigmoid()
    # y=m(y)
    # print(y)
    # zero = torch.zeros_like(y)
    # one = torch.ones_like(y)
    # # a中大于0.5的用one(1)替换,否则a替换,即不变
    # y = torch.where(y >= 0.5, one, y)
    # # a中小于0.5的用zero(0)替换,否则a替换,即不变
    # y = torch.where(y < 0.5, zero, y)
    # print(y)
    # # print(x)
    # # print(x.shape)
    # # print(y)
    # # print(y.shape)





# import math

# import torch
# import torch.nn as nn
# from sympy.printing.tests.test_tensorflow import tf
# from torch import Tensor
# from torch.hub import load_state_dict_from_url
# from torch.nn import functional as F

# from nets.inception_resnetv1 import InceptionResnetV1
# from nets.mobilenet import MobileNetV1, watermark_MobileNetV1
# import nets.model as md
# import numpy as np
# from utils.utils_metrics import evaluate

# def linear(inp,oup):
#     return nn.Sequential(
#         nn.Linear(inp,oup),
#         nn.BatchNorm1d(oup),
#         nn.ReLU()
#     )
# class mobilenet(nn.Module):
#     def __init__(self, pretrained):
#         super(mobilenet, self).__init__()
#         self.model = watermark_MobileNetV1()
#         if pretrained:
#             state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_data",
#                                                 progress=True)
#             self.model.load_state_dict(state_dict)

#         del self.model.fc
#         del self.model.avg

#     def forward(self, x,watermark_in):
#         x = self.model.stage1(x,watermark_in)
#         x = self.model.stage2(x,watermark_in)
#         x = self.model.stage3(x,watermark_in)
#         return x

# class inception_resnet(nn.Module):
#     def __init__(self, pretrained):
#         super(inception_resnet, self).__init__()
#         self.model = InceptionResnetV1()
#         if pretrained:
#             state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_data",
#                                                 progress=True)
#             self.model.load_state_dict(state_dict)

#     def forward(self, x):
#         x = self.model.conv2d_1a(x)
#         x = self.model.conv2d_2a(x)
#         x = self.model.conv2d_2b(x)
#         x = self.model.maxpool_3a(x)
#         x = self.model.conv2d_3b(x)
#         x = self.model.conv2d_4a(x)
#         x = self.model.conv2d_4b(x)
#         x = self.model.repeat_1(x)
#         x = self.model.mixed_6a(x)
#         x = self.model.repeat_2(x)
#         x = self.model.mixed_7a(x)
#         x = self.model.repeat_3(x)
#         x = self.model.block8(x)
#         return x

# class E_watermark(nn.Module):
#     def __init__(self,watermark_size=32, style_dim=512, n_mlp=8, lr_mlp=0.01):
#         super(E_watermark, self).__init__()
#         # layers = [md.PixelNorm()]

#         layers = []
#         layer1 = md.EqualLinear(
#                     watermark_size, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
#                 )
#         layers.append(layer1)


#         for i in range(n_mlp -1):
#             layers.append(
#                 md.EqualLinear(
#                     style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
#                 )
#             )
#         self.style = nn.Sequential(*layers)
#     def forward(self,watermark_in):

#         watermark_out = self.style(watermark_in)
#         return watermark_out

# class D_watermark(nn.Module):
#     def __init__(self,watermark_size=32,lr_mlp=0.01):
#         super(D_watermark,self).__init__()
#         self.stage=nn.Sequential(
#             linear(1024,512),
#             linear(512,256),
#             linear(256,128),
#             linear(128,64),
#             nn.Linear(64,watermark_size)
#         )
#         # layers =[]
#         # in_dim =1024
#         # out_dim=512
#         # for i in range(3):
#         #     layers.append(
#         #         md.EqualLinear(
#         #             in_dim, out_dim, lr_mul=lr_mlp, activation="fused_lrelu"
#         #         )
#         #     )
#         #     in_dim =int(out_dim)
#         #     out_dim=int(out_dim/2)
#         # layers.append(
#         #     nn.Linear(in_dim,watermark_size)
#         # )
#         # self.style=nn.Sequential(*layers)
#     def forward(self,feature_in):
#         watermark_out=self.stage(feature_in)######
#         #watermark_out=self.style(feature_in)
#         return watermark_out

# class Discriminator(nn.Module):
#     def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()

#         channels = {
#             4: 512,
#             8: 512,
#             16: 512,
#             32: 512,
#             64: 256 * channel_multiplier,
#             128: 128 * channel_multiplier,
#             256: 64 * channel_multiplier,
#             512: 32 * channel_multiplier,
#             1024: 16 * channel_multiplier,
#         }

#         convs = [md.ConvLayer(3, channels[size], 1)]

#         log_size = int(math.log(size, 2))

#         in_channel = channels[size]

#         for i in range(log_size, 2, -1):
#             out_channel = channels[2 ** (i - 1)]

#             convs.append(md.ResBlock(in_channel, out_channel, blur_kernel))

#             in_channel = out_channel

#         self.convs = nn.Sequential(*convs)

#         self.stddev_group = 4
#         self.stddev_feat = 1

#         self.final_conv = md.ConvLayer(in_channel + 1, channels[4], 3)
#         self.final_linear = nn.Sequential(
#             md.EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
#             md.EqualLinear(channels[4], 1),
#         )

#     def forward(self, input):
#         out = self.convs(input)

#         batch, channel, height, width = out.shape
#         group = min(batch, self.stddev_group)
#         stddev = out.view(
#             group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
#         )
#         stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
#         stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
#         stddev = stddev.repeat(group, 1, height, width)
#         out = torch.cat([out, stddev], 1)

#         out = self.final_conv(out)

#         out = out.view(batch, -1)
#         out = self.final_linear(out)

#         return out


# class Facenet(nn.Module):
#     def __init__(self,backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False,watermark_size=32):
#         super(Facenet, self).__init__()

#         if backbone == "mobilenet":
#             self.backbone =watermark_MobileNetV1(watermark_size=watermark_size)
#             flat_shape = 1024
#         elif backbone == "inception_resnetv1":
#             self.backbone = inception_resnet(pretrained)
#             flat_shape = 1792
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))

#         self.watermark_Encoder=E_watermark(watermark_size=watermark_size)
#         self.watermark_Decoder=D_watermark(watermark_size=watermark_size)
#         self.avg        = nn.AdaptiveAvgPool2d((1,1))
#         self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
#         self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
#         self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
#         if mode == "train":
#             self.classifier = nn.Linear(embedding_size, num_classes)

#     def forward(self, x,watermark_in,mode = "predict"):
#         if mode == 'predict':
#             watermark_out = self.watermark_Encoder(watermark_in)
#             x = self.backbone(x, watermark_out)
#             x = self.avg(x)
#             x = x.view(x.size(0), -1)
#             watermark_fin=self.watermark_Decoder(x)
#             x = self.Dropout(x)
#             x = self.Bottleneck(x)
#             x = self.last_bn(x)
#             x = F.normalize(x, p=2, dim=1)
#             return x,watermark_fin

#         watermark_out=self.watermark_Encoder(watermark_in)
#         x = self.backbone(x,watermark_out)
#         #x(5,5,1024)
#         x = self.avg(x)
#         #x(1,1,1024)
#         x = x.view(x.size(0), -1)
#         #(batch,1024)
#         watermark_fin = self.watermark_Decoder(x)
#         x = self.Dropout(x)
#         x = self.Bottleneck(x)
#         before_normalize = self.last_bn(x)#feature
#         #add Decoder

#         x = F.normalize(before_normalize, p=2, dim=1)
#         cls = self.classifier(before_normalize)
#         return x, cls,watermark_fin

#     def forward_feature(self, x,watermark_in):
#         watermark_out = self.watermark_Encoder(watermark_in)
#         x = self.backbone(x, watermark_out)
#         #x(5,5,1024)
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         watermark_fin = self.watermark_Decoder(x)
#         x = self.Dropout(x)
#         x = self.Bottleneck(x)
#         before_normalize = self.last_bn(x)
#         x = F.normalize(before_normalize, p=2, dim=1)
#         return before_normalize, x,watermark_in

#     def forward_classifier(self, x):
#         x = self.classifier(x)
#         return x



# import pytorch_lightning as pl

# class LitFaceNet(pl.LightningModule):
#     def __init__(self, model, batch_size, loss, loss2,optimizer, scheduler, watermark_size):
#         super().__init__()
#         self.model = model
#         self.loss = loss
#         self.loss2  = loss2
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.watermark_size = watermark_size
#         self.batch_size = batch_size

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         # it is independent of forward
        
         
#         images, labels = batch
#         watermark = torch.empty(self.batch_size*3, self.watermark_size).uniform_(0, 1)
#         watermark_in = torch.bernoulli(watermark).to(images.device)
#         watermark_in_duplicate = torch.bernoulli(watermark).to(images.device)
        
    
#         outputs1, outputs2,outputs3= self.model(images,watermark_in, "train")

#         # add L_const loss for face embedding
#         outputs1_duplicate, outputs2_duplicate,outputs3_duplicate = self.model(images,watermark_in_duplicate, "train")

#         _watermark_loss = self.loss2(outputs3,watermark_in)
#         _triplet_loss   = self.loss(outputs1, self.batch_size)
#         _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
        
#         #square loss of outputs1 and outputs1_duplicate
#         _constLoss = torch.mean((outputs1 - outputs1_duplicate)**2)

#         _loss           = _triplet_loss + _CE_loss+ _watermark_loss 
#         # _loss           = _triplet_loss + _CE_loss+ _watermark_loss + _constLoss
        
#         self.log('train_watermark_loss', _watermark_loss, sync_dist=True)
#         self.log('train_triplet_loss', _triplet_loss, sync_dist=True)
#         self.log('train_CE_loss', _CE_loss, sync_dist=True)
#         self.log('train_loss', _loss, sync_dist=True)
        
#         with torch.no_grad():
#             m = torch.nn.Sigmoid()
#             y = m(outputs3)
#             zero = torch.zeros_like(y)
#             one = torch.ones_like(y)
#             y = torch.where(y >= 0.5, one, y)
#             y = torch.where(y < 0.5, zero, y)
#             wm_accuracy= torch.mean((y == watermark_in).type(torch.FloatTensor))
#             face_accuracy         = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

#         self.log("train_wm_accuracy", wm_accuracy, sync_dist=True) 
#         self.log("train_face_accuracy", face_accuracy, sync_dist=True) 
        
#         return _loss

#     def validation_step(self, batch, batch_idx):
        
#         images, labels = batch
#         watermark = torch.empty(self.batch_size*3, self.watermark_size).uniform_(0, 1)
#         watermark_in = torch.bernoulli(watermark).to(images.device)
        
    
#         outputs1, outputs2,outputs3= self.model(images,watermark_in, "train")

#         _watermark_loss = self.loss2(outputs3,watermark_in)
#         _triplet_loss   = self.loss(outputs1, self.batch_size)
#         _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
#         _loss           = _triplet_loss + _CE_loss+ _watermark_loss
        
#         self.log('val_watermark_loss', _watermark_loss, sync_dist=True)
#         self.log('val_triplet_loss', _triplet_loss, sync_dist=True)
#         self.log('val_CE_loss', _CE_loss, sync_dist=True)
#         self.log('val_loss', _loss, sync_dist=True)
        
#         m = torch.nn.Sigmoid()
#         y = m(outputs3)
#         zero = torch.zeros_like(y)
#         one = torch.ones_like(y)
#         y = torch.where(y >= 0.5, one, y)
#         y = torch.where(y < 0.5, zero, y)
#         wm_accuracy= torch.mean((y == watermark_in).type(torch.FloatTensor))
#         face_accuracy         = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

#         self.log("val_wm_accuracy", wm_accuracy, sync_dist=True) 
#         self.log("val_face_accuracy", face_accuracy, sync_dist=True) 
    
#     def test_step(self, batch, batch_idx):

#         _, (data_a, data_p, label) = batch 
#         data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
#         watermark_a = torch.empty(data_a.shape[0], self.watermark_size).uniform_(0, 1)
#         watermark_ain = torch.bernoulli(watermark_a)
#         watermark_p = torch.empty(data_p.shape[0], self.watermark_size).uniform_(0, 1)
#         watermark_pin = torch.bernoulli(watermark_p)
#         out_a,out_wa=self.model(data_a,watermark_ain)
#         out_p,out_wp=self.model(data_p,watermark_pin)
#         dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
#         # distances.append(dists.data.cpu().numpy())
#         labels.append(label.data.cpu().numpy())
        
#         labels      = np.array([sublabel for label in labels for sublabel in label])
#         distances   = np.array([subdist for dist in distances for subdist in dist])
#         _, _, accuracy, _, _, _, _ = evaluate(distances,labels)
        
#         self.log("test_face_accuracy", accuracy, sync_dist=True)
       
#     def configure_optimizers(self):
#         # optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return [self.optimizer], [self.scheduler]









# import torch.nn as nn
# from torch.hub import load_state_dict_from_url
# from torch.nn import functional as F

# from nets.inception_resnetv1 import InceptionResnetV1
# from nets.mobilenet import MobileNetV1


# class mobilenet(nn.Module):
#     def __init__(self, pretrained):
#         super(mobilenet, self).__init__()
#         self.model = MobileNetV1()
#         if pretrained:
#             state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_data",
#                                                 progress=True)
#             self.model.load_state_dict(state_dict)

#         del self.model.fc
#         del self.model.avg

#     def forward(self, x):
#         x = self.model.stage1(x)
#         x = self.model.stage2(x)
#         x = self.model.stage3(x)
#         return x

# class inception_resnet(nn.Module):
#     def __init__(self, pretrained):
#         super(inception_resnet, self).__init__()
#         self.model = InceptionResnetV1()
#         if pretrained:
#             state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_data",
#                                                 progress=True)
#             self.model.load_state_dict(state_dict)

#     def forward(self, x):
#         x = self.model.conv2d_1a(x)
#         x = self.model.conv2d_2a(x)
#         x = self.model.conv2d_2b(x)
#         x = self.model.maxpool_3a(x)
#         x = self.model.conv2d_3b(x)
#         x = self.model.conv2d_4a(x)
#         x = self.model.conv2d_4b(x)
#         x = self.model.repeat_1(x)
#         x = self.model.mixed_6a(x)
#         x = self.model.repeat_2(x)
#         x = self.model.mixed_7a(x)
#         x = self.model.repeat_3(x)
#         x = self.model.block8(x)
#         return x
        
# class Facenet_ori(nn.Module):
#     def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False):
#         super(Facenet_ori, self).__init__()
#         if backbone == "mobilenet":
#             self.backbone = mobilenet(pretrained)
#             flat_shape = 1024
#         elif backbone == "inception_resnetv1":
#             self.backbone = inception_resnet(pretrained)
#             flat_shape = 1792
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
#         self.avg        = nn.AdaptiveAvgPool2d((1,1))
#         self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
#         self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
#         self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
#         if mode == "train":
#             self.classifier = nn.Linear(embedding_size, num_classes)

#     def forward(self, x, mode = "predict"):
#         if mode == 'predict':
#             x = self.backbone(x)
#             x = self.avg(x)
#             x = x.view(x.size(0), -1)
#             x = self.Dropout(x)
#             x = self.Bottleneck(x)
#             x = self.last_bn(x)
#             x = F.normalize(x, p=2, dim=1)
#             return x
#         x = self.backbone(x)
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         x = self.Dropout(x)
#         x = self.Bottleneck(x)
#         before_normalize = self.last_bn(x)
        
#         x = F.normalize(before_normalize, p=2, dim=1)
#         cls = self.classifier(before_normalize)
#         return x, cls

#     def forward_feature(self, x):
#         x = self.backbone(x)
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         x = self.Dropout(x)
#         x = self.Bottleneck(x)
#         before_normalize = self.last_bn(x)
#         x = F.normalize(before_normalize, p=2, dim=1)
#         return before_normalize, x

#     def forward_classifier(self, x):
#         x = self.classifier(x)
#         return x



#if __name__ == '__main__':
    # a = torch.empty(3, 32).uniform_(0, 1)
    # a=torch.bernoulli(a)
    # x=torch.rand(3,3,160,160)
    # facenet=Facenet(backbone="mobilenet",num_classes=2,pretrained=False)
    # x,y=facenet(x,a)
    # m=torch.nn.Sigmoid()
    # y=m(y)
    # print(y)
    # zero = torch.zeros_like(y)
    # one = torch.ones_like(y)
    # # a中大于0.5的用one(1)替换,否则a替换,即不变
    # y = torch.where(y >= 0.5, one, y)
    # # a中小于0.5的用zero(0)替换,否则a替换,即不变
    # y = torch.where(y < 0.5, zero, y)
    # print(y)
    # # print(x)
    # # print(x.shape)
    # # print(y)
    # # print(y.shape)





