import torch
import torch.nn as nn
import numpy as np
from conf import device, length, learning_rate, weight_decay


class Pad(nn.Module):

    def __init__(self, offset=0):
        super(Pad, self).__init__()
        self.offset = offset

    def forward(self, x, offset=None):
        # Crop if offset is defined
        if offset is None:
            offset = self.offset
        if offset != 0:
            x = x[..., offset:-offset, offset:-offset]

        # Left-right padding
        x = torch.cat([
            x[..., -1:],
            x,
            x[..., 0:1]
        ], axis=-1)

        # Top-bottom padding
        x = torch.cat([
            torch.zeros(list(x.shape[:-2]) + [1, x.shape[-1]] , device=x.device), # bottom padding
            x, 
            x[..., -1:, :].flip(dims=(-1,)) # top padding
        ], axis=-2)
        
        return x


class Conv(nn.Module):

    def __init__(self, inp, out, mask):
        super(Conv, self).__init__()
        #self.mask = mask[None, None, ...]
        self.pad = Pad(offset=0)
        self.conv = nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.SELU()

    def forward(self, x):
        x = self.pad(x)
        x = self.relu(self.conv(x))
        #x = x * self.mask
        return x


class PCAHRMESModel(nn.Module):

    def __init__(self):
        super(PCAHRMESModel, self).__init()
        self.rnn = nn.GRU(50, 50, batch_first=True, num_layers=3, bidirectional=True)
        self.mlp = nn.Linear(100, 50)

    def forward(self, x):
        x = torch.nan_to_num(x).requires_grad_()
        x = self.rnn(x)[0][:, -1, :]
        x = self.mlp(x)
        return x


class HRMESModel(nn.Module):

    def __init__(self, bathy, mask=None):
        super(HRMESModel, self).__init__()
        self.mask = mask
        self.bathy = torch.from_numpy(bathy.to_numpy()).to(device).float()[None, None]
        self.layers_g1 = nn.Sequential(
            Pad(offset=1),
            nn.Conv2d(length+1, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SELU()
        )
        self.layers_g2 = nn.Sequential(
            #Conv(length, 32, mask=mask),
            Conv(32, 64, mask=mask),
            Conv(64, 128, mask=mask),
            Conv(128, 64, mask=mask),
            Conv(64, 32, mask=mask),
            Pad(offset=0),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0, bias=True)
            #Conv(32, 1, mask=mask)
            #nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = torch.nan_to_num(x).requires_grad_()
        x = torch.cat(
            [x, self.bathy.expand(x.shape[0], 1, -1, -1)],
            axis=1
        )
        x = self.layers_g1(x)
        x = self.layers_g2(x)
        x = x.squeeze(dim=1)
        return x


def define_model_optimizer_criterion(bathy, pca_model=False):
    if pca_model:
        model = PCAHRMESModel().to(device)
    else:
        model = HRMESModel(bathy).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model has {params} parameters")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, criterion, optimizer


def compute_loss(criterion, output, target, bool_mask=None):
    if bool_mask is None:
        return criterion(output, target)
    else:
        target = Pad.forward(..., target, offset=1)
        return criterion(
            torch.masked_select(output, bool_mask),
            torch.masked_select(target, bool_mask)
        )