
import torch

def loss(recons):
    recons_mean = torch.mean(recons, dim=1, keepdim=True)

    # This is 0 if every slot produces same output --> unwanted
    recons_mean_deviance = torch.abs(recons - recons_mean)

    # Take the sum of pixel deviances
    recons_mean_deviance_sum = recons_mean_deviance.sum()

    # Normalize
    recons_mean_deviance_norm = recons_mean_deviance_sum / recons.shape.numel()

    # Make 0 the target:
    recons_mean_similarity = 1 - recons_mean_deviance_norm

    return recons_mean_similarity

recons = torch.ones((10,10,1,84,84))

print(loss(recons))