import torch
import torch.nn.functional as F 

eps = 1e-8

class FidelityLoss(torch.nn.Module):
    def __init__(self):
        super(FidelityLoss, self).__init__()

    def _pcal(self, y1, y1_var, y2, y2_var):
        y_diff = (y1 - y2)
        y_var = y1_var + y2_var 
        p = 0.5 * (1 + torch.erf(y_diff / torch.sqrt(2 * y_var)))
        return p

    def forward(self, y1, y1_var, y2, y2_var, g):
        g = g.view(-1, 1)
        p = self._pcal(y1, y1_var, y2, y2_var).view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
        return loss.mean()