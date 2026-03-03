import torch
import torch.nn as nn

class PRetLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.delta = float(configs.delta)
        self.risk_param = float(configs.risk_param)

    def forward(self, w, y, cov):
        pret = (w * y).sum(dim=-1)            
        risk = torch.einsum('bn,bnm,bm->b', w, cov, w)  
        pret_term = pret.mean()                
        risk_term = risk.mean()                
        loss = -pret_term + self.risk_param * self.delta * risk_term
        return loss

