import os
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def show_mnist_images(data: np.ndarray) -> None:
    num_images = len(data)

    fig, ax = plt.subplots(1, num_images)

    if num_images == 1:
        image = np.array(data[0])
        ax.imshow(image, cmap="Greys")
    else:
        for a, image in zip(ax, data):
            image = np.array(image)
            a.imshow(image, cmap="Greys")
    
    plt.show()


def save_images(out, writer: SummaryWriter, global_step: int) -> None:
    images = {
        "Orig": out["input"],
        "Recon": out["decoded"],
        "Noisy_Input": out["noisy"]
    }

    for key in images:
        fig = plt.figure()
        axes = plt.axes()
        axes.imshow(torch.permute(images[key][0], (1,2,0)).cpu().detach().numpy())
        writer.add_figure(tag=f"Sample/{key}", figure=fig, global_step=global_step)
    
    # Special case: discrete tensor
    discrete = out["discrete"][0].cpu().detach().numpy()
    discrete = np.concatenate([discrete[::2], discrete[1::2]])
    fig = plt.figure()
    axes = plt.axes()
    axes.imshow(discrete.reshape((discrete.shape[0] // 8, 8)), cmap="Greys")
    writer.add_figure(tag=f"Sample/Discrete", figure=fig, global_step=global_step)


def set_manual_seed(seed: int = 1):
    """Set the seed for the PRNGs."""
    os.environ['PYTHONASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.benchmark = True
