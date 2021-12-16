import torch
import torch.nn as nn
import torch.nn.functional as F

'''
All loss functions found here
'''

# Gumbel Softmax loss - formula provided in section 3.1.6 Gumbel Softmax
def gs_loss(logit_q, logit_p, eps=1e-20):
    q = F.softmax(logit_q, dim=-1)
    q = q / torch.sum(q, dim=-1, keepdim=True)
    p = F.softmax(logit_p, dim=-1)
    p = p / torch.sum(p, dim=-1, keepdim=True)
    log_q = torch.log(q + eps)
    log_p = torch.log(p + eps)
    loss = q * (log_q - log_p)
    loss_sum = torch.sum(loss)
    return loss_sum


# Binary Concrete loss - formula provided in section 3.1.7 Binary Concrete
# logit_p for logits (network outputs before activation), p is for the Bernoulli(p) prior
def bc_loss(logit_q, logit_p=None, p=None, eps=1e-20):
    if logit_p is None and p is None:
        raise ValueError("Both logit_p and p cannot be None")
    elif p is None:
        p = torch.sigmoid(logit_p)
    q = torch.sigmoid(logit_q)
    log_q0 = torch.log(q + eps)
    log_q1 = torch.log(1 - q + eps)
    log_p0 = torch.log(p + eps)
    log_p1 = torch.log(1 - p + eps)
    loss = q * (log_q0 - log_p0) + (1 - q) * (log_q1 - log_p1)
    loss_sum = torch.sum(loss)
    return loss_sum


# Follows equations given in section 3.1.8 Loss Functions
def total_loss(out, p, beta_z, losses=None):

    # KL losses
    z0_prior = bc_loss(out["encoded"], p=p)

    # Reconstruction losses
    criterion = nn.MSELoss()

    x0_recon = criterion(out["input"], out["decoded"])

    # Store losses for future plotting
    if losses is not None:
        losses['z0_prior'].append(z0_prior.detach().cpu().numpy())
        losses['x0_recon'].append(x0_recon.detach().cpu().numpy())
    
    # Follows formulas provided in paper
    loss = beta_z * z0_prior + x0_recon

    return loss, losses