import torch
from torch import nn
from torch.nn import functional as F


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0., reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target - 1, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)


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
        input = torch.sigmoid(input)
        input = torch.clamp(input, 1e-10, 1.-1e-10)
        c0 = input ** self.gamma
        c1 = (1. - input) ** self.gamma
        return torch.mean(- (c1 * target * torch.log(input) + c0 * (1 - target) * torch.log(1. - input)))
