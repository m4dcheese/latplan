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
        axes.imshow(images[key][0].permute((1,2,0)).cpu().detach().numpy())
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

def hungarian_matching(attrs: torch.Tensor, preds_attrs: torch.Tensor, verbose=0):
    """
    Receives unordered predicted set and orders this to match the nearest GT set.
    :param attrs:
    :param preds_attrs:
    :param verbose:
    :return:
    """
    assert attrs.shape[1] == preds_attrs.shape[1]
    assert attrs.shape == preds_attrs.shape
    from scipy.optimize import linear_sum_assignment
    matched_preds_attrs = preds_attrs.clone()
    idx_mappings = []
    for sample_id in range(attrs.shape[0]):
        # using euclidean distance
        cost_matrix = torch.cdist(attrs[sample_id], preds_attrs[sample_id]).detach().cpu()

        idx_mapping = linear_sum_assignment(cost_matrix)
        # convert to tuples of [(row_id, col_id)] of the cost matrix
        idx_mapping = [(idx_mapping[0][i], idx_mapping[1][i]) for i in range(len(idx_mapping[0]))]
        idx_mappings.append(idx_mapping)

        for i, (row_id, col_id) in enumerate(idx_mapping):
            matched_preds_attrs[sample_id, row_id, :] = preds_attrs[sample_id, col_id, :]
        if verbose:
            print('GT: {}'.format(attrs[sample_id]))
            print('Pred: {}'.format(preds_attrs[sample_id]))
            print('Cost Matrix: {}'.format(cost_matrix))
            print('idx mapping: {}'.format(idx_mapping))
            print('Matched Pred: {}'.format(matched_preds_attrs[sample_id]))
            print('\n')
            # exit()

    return matched_preds_attrs, idx_mappings