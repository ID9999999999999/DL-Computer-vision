"""Local read-only PyTorch compatibility adapter for YOLOv8n-cls checkpoints.
This is not the Ultralytics package: use to evaluate checkpoints when unavailable.
"""
import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2f(nn.Module):
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Classify(nn.Module):
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        # Ultralytics classification head: conv -> avgpool -> flatten -> dropout -> linear
        x = self.conv(x)
        return self.linear(self.drop(self.pool(x).flatten(1)))

class ClassificationModel(nn.Module):
    def forward(self, x):
        return self.model(x)

SAFE = [
    (ClassificationModel, 'ultralytics.nn.tasks.ClassificationModel'),
    (Conv,'ultralytics.nn.modules.conv.Conv'),
    (Bottleneck,'ultralytics.nn.modules.block.Bottleneck'),
    (C2f,'ultralytics.nn.modules.block.C2f'),
    (Classify,'ultralytics.nn.modules.head.Classify'),
    T.Compose, T.Resize, T.CenterCrop, T.ToTensor, T.Normalize,
    InterpolationMode,
    nn.Sequential, nn.Conv2d, nn.BatchNorm2d, nn.SiLU,
    nn.ModuleList, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Linear,
]

def load_checkpoint(file):
    # Explicit weights_only with allowlisted class objects, no general pickle execution
    with torch.serialization.safe_globals(SAFE):
        ckpt = torch.load(file, map_location='cpu', weights_only=True)
    return ckpt
