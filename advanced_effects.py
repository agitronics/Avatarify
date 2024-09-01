import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

class DeepDreamEffect(nn.Module):
    def __init__(self):
        super(DeepDreamEffect, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x, layer_num=5, iterations=20, lr=0.01):
        x = self.transform(x).unsqueeze(0)
        for _ in range(iterations):
            x = x.requires_grad_()
            output = self.model(x)
            loss = output[0, layer_num].mean()
            loss.backward()
            x = x + lr * x.grad.data
            x = x.detach()
        
        x = x.squeeze().permute(1, 2, 0).cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        return (x * 255).astype(np.uint8)

class CartoonEffect:
    def __init__(self):
        self.edge_preserve_filter = cv2.edgePreservingFilter(flags=cv2.RECURS_FILTER)

    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = self.edge_preserve_filter(frame)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

class GlitchEffect:
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def apply(self, frame):
        h, w = frame.shape[:2]
        shift = int(w * self.intensity)
        
        for _ in range(10):  # Apply 10 random glitches
            y1 = np.random.randint(h)
            y2 = np.random.randint(y1, h)
            
            tmp = frame[y1:y2, :shift].copy()
            frame[y1:y2, :w-shift] = frame[y1:y2, shift:]
            frame[y1:y2, w-shift:] = tmp
        
        return frame

class VHSEffect:
    def __init__(self):
        self.noise = np.zeros((720, 1280), dtype=np.uint8)

    def apply(self, frame):
        # Add color bleeding
        frame = cv2.blur(frame, (3, 3))
        
        # Add noise
        self.noise = np.random.randint(0, 50, self.noise.shape, dtype=np.uint8)
        frame = cv2.add(frame, np.stack([self.noise]*3, axis=-1))
        
        # Add horizontal lines
        lines = np.random.randint(0, 5, frame.shape[0]) * 255
        frame += np.stack([lines]*3, axis=-1)
        
        # Add vertical color shift
        frame[:, :, 0] = np.roll(frame[:, :, 0], 2, axis=1)
        frame[:, :, 2] = np.roll(frame[:, :, 2], -2, axis=1)
        
        return frame

class RainbowEffect:
    def __init__(self):
        self.hue = 0

    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.hue) % 180
        self.hue = (self.hue + 1) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class AdvancedVideoEffects:
    def __init__(self):
        self.deep_dream = DeepDreamEffect()
        self.cartoon = CartoonEffect()
        self.glitch = GlitchEffect()
        self.vhs = VHSEffect()
        self.rainbow = RainbowEffect()

    def apply_effect(self, frame, effect_name):
        if effect_name == "deep_dream":
            return self.deep_dream(frame)
        elif effect_name == "cartoon":
            return self.cartoon.apply(frame)
        elif effect_name == "glitch":
            return self.glitch.apply(frame)
        elif effect_name == "vhs":
            return self.vhs.apply(frame)
        elif effect_name == "rainbow":
            return self.rainbow.apply(frame)
        else:
            return frame