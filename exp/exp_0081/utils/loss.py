import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSELoss = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.MSELoss(input, target))


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = torch.clamp(input, 1e-10, 1.-1e-10)
        return torch.mean((- (target * torch.log(input) + (1 - target) * torch.log(1. - input))))


class FOCALLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        c0 = input ** self.gamma
        c1 = (1. - input) ** self.gamma
        input = torch.sigmoid(input)
        input = torch.clamp(input, 1e-10, 1.-1e-10)
        return torch.mean(- (c1 * target * torch.log(input) + c0 * (1 - target) * torch.log(1. - input)))
