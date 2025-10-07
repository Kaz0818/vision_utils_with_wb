from torch import nn 
from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf

class SimpleCNN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 30),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(30, out_ch),
        )
        
    def forward(self, x):
        x = self.features(x)
        out =self.classifier(x)
        return out
    
    
