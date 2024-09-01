import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AdvancedFaceSwapNetwork(nn.Module):
    def __init__(self):
        super(AdvancedFaceSwapNetwork, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, source, target):
        source_features = self.encoder(self.face_pool(source))
        target_features = self.encoder(self.face_pool(target))
        
        # Implement attention mechanism
        attention = F.softmax(torch.bmm(source_features.view(source_features.size(0), -1, 256), 
                                        target_features.view(target_features.size(0), 256, -1)), dim=2)
        out = torch.bmm(attention, target_features.view(target_features.size(0), 256, -1))
        out = out.view(out.size(0), 2048, 8, 8)
        
        return self.decoder(out)

def load_face_swap_model(path):
    model = AdvancedFaceSwapNetwork()
    model.load_state_dict(torch.load(path))
    return model