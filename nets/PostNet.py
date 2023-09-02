
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
class Postnet(nn.Module):
    def __init__(self,watermark_size=32):
        super(Postnet,self).__init__()
        self.watermark_size=watermark_size
        self.Encoder = nn.Sequential(
            linear(128+self.watermark_size,256),
            linear(256,512),
            linear(512,1024),
            linear(1024,512),
            linear(512,256),
            nn.Linear(256,128),

        )
        self.Decoder = nn.Sequential(
            linear(128,64),
            linear(64,32),
            nn.Linear(32,32)
        )

    def forward(self,x,watermark_in):
        x=torch.cat((x,watermark_in),dim=1)
        x = self.Encoder(x)
        embedding =x
        x = self.Decoder(x)
        watermark_out = x
        return embedding,watermark_out