import torch
import torch.nn as nn
import torch.nn.functional as F

class MapEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_size, kernels, strides):
        super(MapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        output_convs_size = torch.ones(input_size).unsqueeze(0)

        for i, hidden_channels in enumerate(hidden_channels):
            self.convs.append(nn.Conv2d(input_channels if i==0 else hidden_channels[i-1], hidden_channels[i],
                              kernel_size=kernels[i], stride=strides[i]))
            
            output_convs_size = self.convs[i](output_convs_size)

        self.fc = nn.Linear(output_convs_size.numel(), output_size)

    def forward(self, maps):
        for conv in self.convs:
            x = F.leaky_relu(conv(mmapsap), 0.2)
        x = torch.flatten(x, start_dim=1)
        maps_enc = self.fc(x)
        return maps_enc