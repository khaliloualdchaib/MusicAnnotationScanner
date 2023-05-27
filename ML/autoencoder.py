import torch
from torch import nn
from math import floor

# BatchNorm2d ?

class Autoencoder(nn.Module):
    
    def __init__(self, input_height, input_width) -> None:
        super().__init__()
        self.height = input_height
        self.width = input_width

        self.encoder_cl = nn.Sequential(
            nn.Conv2d(3,2,3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(2,4,3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(4,8,3,stride=2, padding=0),
            nn.ReLU(True),
        )

        self.decoder_cl = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, output_padding=1)
        )
        
    def calculate_fc_input(self):
        width = self.width
        height = self.height
        lastfilter_size = 0
        for module in self.encoder_cl.children():
            if isinstance(module,nn.Conv2d):
                height = floor((height + 2* module.padding[0] - module.kernel_size[0]) / module.stride[0]) + 1
                width = floor((width + 2* module.padding[1] - module.kernel_size[1]) / module.stride[1]) + 1
            elif isinstance(module,nn.MaxPool2d):
                height = floor((height - module.kernel_size) / module.stride) + 1
                width = floor((width - module.kernel_size) / module.stride) + 1
                continue
            elif isinstance(module,nn.ReLU) or isinstance(module, nn.Dropout2d):
                continue
            elif isinstance(module,nn.BatchNorm2d):
                continue
            lastfilter_size = module.out_channels
        return lastfilter_size, height, width

    # Forward
    def forward(self, x):
        x = self.encoder_cl(x)
        x = self.decoder_cl(x)
        x = torch.sigmoid(x)
        return x