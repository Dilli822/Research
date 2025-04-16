import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution Block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        # Pointwise convolution (1x1 convolution)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

# Simple MobileNetV1 Model
class SimpleMobileNetV1(nn.Module):
    def __init__(self, num_classes=10):  # Example: for 10-class classification
        super(SimpleMobileNetV1, self).__init__()
        
        # Initial Conv layer
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise Separable Convolutions (MobileNetV1 layers)
        self.block1 = DepthwiseSeparableConv(32, 64, stride=1)
        self.block2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.block3 = DepthwiseSeparableConv(128, 128, stride=1)
        self.block4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.block5 = DepthwiseSeparableConv(256, 256, stride=1)
        self.block6 = DepthwiseSeparableConv(256, 512, stride=2)
        
        # 5 more depthwise separable blocks
        self.block7 = DepthwiseSeparableConv(512, 512, stride=1)
        self.block8 = DepthwiseSeparableConv(512, 512, stride=1)
        self.block9 = DepthwiseSeparableConv(512, 512, stride=1)
        self.block10 = DepthwiseSeparableConv(512, 512, stride=1)
        self.block11 = DepthwiseSeparableConv(512, 1024, stride=2)
        
        # Fully connected layer
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.initial_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x

# Instantiate the model
model = SimpleMobileNetV1(num_classes=10)  # For a 10-class classification problem

# Print model architecture
print(model)
