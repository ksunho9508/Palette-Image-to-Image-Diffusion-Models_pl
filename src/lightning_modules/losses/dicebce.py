import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.sigmoid()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE = -(
            targets * torch.log(inputs + 1e-6)
            + (1 - targets) * torch.log(1 - inputs + 1e-6)
        ).mean()
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
