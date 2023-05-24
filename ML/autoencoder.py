import torch
from torch import nn
from math import floor

# BatchNorm2d ?

class Autoencoder(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        #Convolutional layers
        self.encoder_cl = nn.Sequential(
            nn.Conv2d(3,8,3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8,16,3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16,32,3,stride=2, padding=0),
            nn.ReLU(True),
        )

        self.decoder_cl = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )

        
        #Flatten lineair
        self.flatten = nn.Flatten(start_dim=1)
        #Unflatten
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(self.calculate_fc_input()[0], self.calculate_fc_input()[1], self.calculate_fc_input()[2]))
        # Calculat fc inputsize
        self.fcinputsize = self.calculate_fc_input()[0] * self.calculate_fc_input()[1] * self.calculate_fc_input()[2]
        #Fully conected layers
        self.enconder_fcl = nn.Sequential(
            nn.Linear(self.fcinputsize,144),
            nn.ReLU(True),
            nn.Linear(144, 72)
        )

        self.decoder_fcl = nn.Sequential(
            nn.Linear(72, 144),
            nn.ReLU(True),
            nn.Linear(144, self.fcinputsize),
            nn.ReLU(True)
        )
        
    # Calculate fc input
    def calculate_fc_input(self):
        width = 500
        height = 500
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
        x = self.flatten(x)
        x = self.enconder_fcl(x)
        x = self.decoder_fcl(x)
        x = self.unflatten(x)
        x = self.decoder_cl(x)
        x = torch.sigmoid(x)
        return x