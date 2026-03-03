import torch
import torch.nn as nn

class CovUpdater(nn.Module):
    def __init__(self, configs):
        super(CovUpdater, self).__init__()
        self.decay = configs.decay
        self.cov = None
        self.cur_cov = None

    def forward(self, x): # x: [B, T, N]
        if self.cov is None:
            self.cov = torch.zeros(x.shape[0], x.shape[-1], x.shape[-1]).to(x.device)
        for i in range(x.shape[0]):
            if self.cur_cov is None:
                self.cur_cov = torch.cov(x[i].transpose(0, 1), correction=0)
            else:
                new_cov = torch.cov(x[i].transpose(0, 1), correction=0)
                self.cur_cov = self.decay * self.cur_cov + (1 - self.decay) * new_cov
            self.cov[i] = self.cur_cov + 1e-6 * torch.eye(x.shape[-1]).to(x.device)
        return self.cur_cov, self.cov # [N, N], [B, N, N]

    def reset(self):
        self.cov = None
        self.cur_cov = None
        

def repeat_cov(cur_cov, cov, horizon):
    out_cur_cov = cur_cov.unsqueeze(0).repeat(horizon, 1, 1)
    out_cov = cov.unsqueeze(1).repeat(1, horizon, 1, 1)
    return out_cur_cov, out_cov # [H, N, N], [B, H, N, N]
    

def update_cov(cur_cov, cov, horizon, x, yhat):
    """
    Update with predicted returns
    """
    x_cat = torch.cat([x, yhat], dim=1) # [B, T+H, N]
    B, T_H, N = x_cat.shape
    T = x.shape[1]
    
    out_cur_cov = torch.zeros(horizon, N, N).to(x.device)
    out_cov = torch.zeros(B, horizon, N, N).to(x.device)
    
    for i in range(B):
        batch_cur_cov = cur_cov.clone()
        for h in range(horizon):
            new_data = x_cat[i, :T+h+1]
            new_cov = torch.cov(new_data.transpose(0, 1), correction=0)
            batch_cur_cov = 0.95 * batch_cur_cov + 0.05 * new_cov
            out_cur_cov[h] = batch_cur_cov + 1e-6 * torch.eye(N).to(x.device)
            out_cov[i, h] = batch_cur_cov + 1e-6 * torch.eye(N).to(x.device)
    
    return out_cur_cov, out_cov # [H, N, N], [B, H, N, N]