import numpy as np
import torch


def rand_bbox(size, l):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1.0 - l)
    cut_W = np.int(W * cut_rat)
    cut_H = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_W // 2, 0, W)
    bbx2 = np.clip(cx + cut_W // 2, 0, W)
    bby1 = np.clip(cy - cut_H // 2, 0, H)
    bby2 = np.clip(cy + cut_H // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    new_data = data.clone()

    l = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    new_data = new_data * l + data[indices, :, :, :] * (1 - l)
    targets = (target, shuffled_target, l)
    return new_data, targets


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    new_data = data.clone()

    l = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), l)
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    l = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) /
             (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, l)

    return new_data, targets
