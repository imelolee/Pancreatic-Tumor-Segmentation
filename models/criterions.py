import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class TverskyLoss3d(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)+
                        0.3 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)

class TverskyLoss2d(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1)+
                        0.3 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)