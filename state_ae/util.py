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
        "Recon": out["decoded"]
    }

    for key in images:
        fig = plt.figure()
        axes = plt.axes()
        axes.imshow(torch.permute(images[key][0], (1,2,0)).cpu().detach().numpy())
        writer.add_figure(tag=f"Sample/{key}", figure=fig, global_step=global_step)