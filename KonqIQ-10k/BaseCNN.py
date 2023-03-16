import torch
import torch.nn as nn
import resnet_prune 
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np
import random


class BaseCNN(nn.Module):
    def __init__(self, config=None):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        self.config = config
        self.backbone = resnet_prune.prune_resnet18(pretrained=True)

        if self.config.fz:
            # Freeze all previous layers.
            for key, param in self.backbone.named_parameters():
                param.requires_grad = False

        outdim = 1
        self.backbone.fc = nn.Linear(512, outdim, bias=True)
        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.backbone.fc.weight.data)
        if self.backbone.fc.bias is not None:
            nn.init.constant_(self.backbone.fc.bias.data, val=0)

    def forward(self, x_init):
        """
        Forward pass of the network.
        """
        x = self.backbone.conv1(x_init)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x).view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        x = self.backbone.fc(x)
        mean = x[:, 0]
        # t = x[:, 1]
        # var = nn.functional.softplus(t)**2
        var = torch.ones_like(mean)**2
        return mean, var
        