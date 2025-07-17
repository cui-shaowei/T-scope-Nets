import torch
import torch.nn as nn
from torchvision import models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features

        # Simplified MLP with one layer
        self.fc_layers = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Forward pass through ResNet18 (frozen layers)
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)

        # Forward pass through MLP
        x = self.fc_layers(x)

        return x
