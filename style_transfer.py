import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class AdvancedStyleTransfer(nn.Module):
    def __init__(self):
        super(AdvancedStyleTransfer, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])
        for param in self.parameters():
            param.requires_grad = False
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def style_transfer(vgg, content, style, alpha=1.0):
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f[-1], style_f[-1])
    return feat * alpha + content_f[-1] * (1 - alpha)

def load_style_transfer_model():
    return AdvancedStyleTransfer()