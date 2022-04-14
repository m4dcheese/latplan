import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parameters = {}

'''
All loss functions found here
'''

def print_if(text, value):
    if (value):
        print(text)

# Gumbel Softmax loss - formula provided in section 3.1.6 Gumbel Softmax
def gs_loss(logit_q, p=.5, eps=1e-20, print_out=False):
    p_true, p_false = p, 1 - p
    log_p_true, log_p_false = np.log(p_true), np.log(p_false)

    logit_q = torch.reshape(logit_q, (logit_q.shape[0], logit_q.shape[1] // 2, 2))
    print_if(logit_q, print_out)
    q = F.softmax(logit_q, dim=-1)
    print_if(q, print_out)
    log_q = torch.log(q + eps)
    print_if(log_q, print_out)
    log_q[:, :, 0] = (log_p_true - log_q[:, :, 0]) * q[:, :, 0]
    log_q[:, :, 1] = (log_p_false - log_q[:, :, 1]) * q[:, :, 1]
    print_if(log_q, print_out)
    loss_sum = torch.sum(log_q, dim=(-2, -1))
    print_if(loss_sum, print_out)
    return - loss_sum.mean()


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


def zero_suppression_loss(logits: torch.Tensor):
    return logits[:, ::2].mean()

def weighted_zero_suppression_loss(logits: torch.Tensor):
    weights = torch.FloatTensor(np.linspace(0, 2, logits[:, ::2].shape[-1])**2).to(logits.device)
    weighted_logits = logits[:, ::2] * weights
    return weighted_logits.mean()

def root_zero_suppression_loss(logits: torch.Tensor):
    logits = ((logits[:, ::2].mean(dim=0) + 1e-8)**.5)
    return logits.mean()

def weighted_root_zero_suppression_loss(logits: torch.Tensor):
    weights = torch.FloatTensor(np.linspace(0.1, 2, logits[:, ::2].shape[-1])**2).to(logits.device)
    weighted_logits = ((logits[:, ::2].mean(dim=0) + 1e-8)**.5) * weights
    return weighted_logits.mean()

def fancy_func_zero_suppression_loss(logits: torch.Tensor):
    variable_means = logits[:, ::2].mean(dim=0)
    logits = (variable_means + 1e-8)**.5 * (variable_means - .5)**2 + .5 * variable_means
    return logits.mean()

def beta_scheduler(step, plan="paper", **kwargs) -> float:
    """
    Returns the correct beta factor for the loss regularization term
    Args:
        step:
            Current training step
        total_steps:
            Total training steps
        plan:
            One of ["paper", "increase", "increase_after_warmup"]. Scheduling plan to use
    """
    iters_per_epoch = int(parameters.total_samples / parameters.batch_size)
    total_steps = parameters.epochs * iters_per_epoch
    if plan == "paper":
        return 0 if step < total_steps / 3 else parameters.beta
    elif plan == "increase":
        increase_end = int(kwargs["fraction_increase_end"] * total_steps)
        if step < increase_end:
            return parameters.beta * step / increase_end
        else:
            return parameters.beta
    elif plan == "increase_after_warmup":
        increase_end = int(kwargs["fraction_increase_end"] * total_steps)
        increase_start = parameters.warm_up_steps
        if step < increase_start:
            return 0
        elif step < increase_end:
            return parameters.beta * (step - increase_start) / (increase_end - increase_start)
        else:
            return parameters.beta


# Follows equations given in section 3.1.8 Loss Functions
def total_loss(out, target, p, beta, step, writer, params):
    global parameters
    parameters = params
    # Reconstruction losses
    criterion = nn.MSELoss()

    recon = criterion(out["decoded"], target)

    # Regularization losses
    kl_loss = gs_loss(out["encoded"], p=p)

    zs_criteria_library = {
        "paper": zero_suppression_loss,
        "weighted": weighted_zero_suppression_loss,
        "root": root_zero_suppression_loss,
        "weighted_root": weighted_root_zero_suppression_loss,
        "fancy": fancy_func_zero_suppression_loss
    }
    zs_criterion = zs_criteria_library[parameters.zero_supp_version]

    zs_loss = zs_criterion(out["discrete"])

    beta = beta_scheduler(step, plan=parameters.loss_beta_plan, **parameters.loss_kwargs)

    writer.add_scalar("hyper/beta", beta, global_step=step)    

    losses = {
        "kl": kl_loss.detach().cpu().numpy(),
        "zs": zs_loss.detach().cpu().numpy(),
        "recon": recon.detach().cpu().numpy(),
    }
    # Follows formulas provided in paper
    loss = beta * zs_loss + recon

    return loss, losses
