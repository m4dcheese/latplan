from typing import Callable
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from model import StateAE
from data import get_loader
from parameters import parameters
from state_ae.slot_attention_state_ae import DiscreteSlotAttention_model

import math

def count_differing_tiles(out1: torch.Tensor, out2: torch.Tensor):
    diff = out1 - out2
    
    plt.imshow(diff[0].permute((1,2,0)).cpu().detach().numpy())
    plt.show()
    plt.imshow(out1[0].permute((1,2,0)).cpu().detach().numpy())
    plt.figure()
    plt.imshow(out2[0].permute((1,2,0)).cpu().detach().numpy())
    plt.show()
    tile_size = math.floor(parameters.image_size[0] / 3)
    tile_borders = [i * tile_size for i in range(4)]

    diff_sums = np.zeros((parameters.batch_size, 3,3))


    for i, (lower_row, upper_row) in enumerate(zip(tile_borders[:-1], tile_borders[1:])):
        for j, (lower_col, upper_col) in enumerate(zip(tile_borders[:-1], tile_borders[1:])):
            diff_sums[:, i, j] = diff[:, :, lower_row:upper_row+1, lower_col:upper_col+1].square().sum(dim=(1,2,3)).cpu().detach().numpy()
    
    print(diff_sums[0])


def affected_tiles_sasae(model: DiscreteSlotAttention_model):
    usecuda = torch.cuda.is_available() and not parameters['no_cuda']
    loader = get_loader(
        dataset="color_shapes",
        image_size=parameters.image_size,
        deletions=parameters.deletions,
        total_samples=parameters.total_samples,
        batch_size=parameters.batch_size,
        usecuda=usecuda
    )

    for i, sample in tqdm(enumerate(loader)):
        img_input = sample[0].to(f"cuda:{parameters.device_ids[0]}")
        img_target = sample[1].to(f"cuda:{parameters.device_ids[0]}")
        img_output, _, _, _, _, discrete = model(img_input, epoch=10000)
        latent_size = int(discrete.shape[-1] / 2)
        # for each sample, select 1 slot and (latent_size / 2) indices where bits get flipped
        idx_mapping = [0,1,2,3,4,5,6,7,11,15]
        random_flip_bit_indices = np.random.randint(0, 10, parameters.batch_size)
        random_flip_bit_indices = np.array([idx_mapping[i] for i in random_flip_bit_indices])
        random_flip_slot_indices = np.random.randint(0, parameters.slots, parameters.batch_size)

        for i in range(parameters.batch_size):
            global_slot_idx = i * parameters.slots + random_flip_slot_indices[i]
            tmp_pos = discrete[global_slot_idx, random_flip_bit_indices[i] * 2]
            tmp_neg = discrete[global_slot_idx, random_flip_bit_indices[i] * 2 + 1]
            discrete[global_slot_idx, random_flip_bit_indices[i] * 2] = 1 - tmp_pos
            discrete[global_slot_idx, random_flip_bit_indices[i] * 2 + 1] = 1 - tmp_neg
        
        img_output_flipped, _, _, _, _, _ = model.forward(img_input, discrete=discrete, epoch=10000)

        diff_tiles = count_differing_tiles(img_output, img_output_flipped)




def evaluate(
    model: torch.nn.Module,
    samples: int = 100,
    adapt_discrete: Callable = lambda x: x,
    batch_size: int = parameters.batch_size,
    latent_space_size: int = parameters.latent_size
):
    usecuda = torch.cuda.is_available() and not parameters['no_cuda']
    loader = get_loader(
        dataset="color_shapes",
        image_size=parameters.image_size,
        deletions=parameters.deletions,
        total_samples=samples,
        batch_size=batch_size,
        usecuda=usecuda
    )

    mse = torch.nn.MSELoss()
    mse_losses = []
    bit_variances = []

    # Effective bits - Tensor to save the sum of all propositions
    discrete_usage = torch.zeros(latent_space_size)
    for i, sample in tqdm(enumerate(loader)):
        # State variance over bits ([1] 6.1)
        discrete_tensors = []
        #recons = []
        #masks = []
        if i < 5:
            for iteration in range(1000):
                img = sample[0].to(f"cuda:{parameters.device_ids[0]}")
                out = model(img, epoch=10000)
                discrete = adapt_discrete(out)
                discrete_tensors.append(discrete.cpu().detach())
                #recons.append(out[1].cpu().detach())
                #masks.append(out[2])
            discrete_tensors = torch.stack(discrete_tensors)
            #recons = torch.stack(recons)
            #masks = torch.stack(masks)
            # for j in range(1, discrete_tensors.shape[0]):
            #     discrete_tensors[j], idx_mapping = hungarian_matching(discrete_tensors[0], discrete_tensors[j])
                # write_discrete(writer, j, discrete_tensors[j])
                # for sample_id in range(1):
                #     idx_mapping_sample = dict(idx_mapping[sample_id])
                #     ordered_slot_recons = recons[j, sample_id, [idx_mapping_sample[k] for k in range(10)]].unsqueeze(0)
                #     ordered_slot_masks = masks[j, sample_id, [idx_mapping_sample[k] for k in range(10)]].unsqueeze(0)
                #     write_slot_imgs(writer, j, ordered_slot_recons.permute(0,1,3,4,2))
                #     print(ordered_slot_masks.shape)
                #     write_mask_imgs(writer, j, ordered_slot_masks.permute(0,1,3,4,2))
            # variance_over_iterations = discrete_tensors.var(dim=0)
            # variance_over_iterations_per_slot = variance_over_iterations.mean(dim=2)
            # variance_over_iterations_per_slot[:, variance_over_iterations_per_slot.argmax(axis=1)] = 0
            # bit_variance = variance_over_iterations_per_slot
            bit_variance = discrete_tensors.var(dim=0)
        else:
            #break
            img = sample[0].to(f"cuda:{parameters.device_ids[0]}")
            out = model(img, epoch=10000)
            discrete = adapt_discrete(out)
        if i == 0:
            plt.axis('off')
            plt.imshow(out["input"][0].permute(1,2,0).cpu().detach().numpy())
            plt.show()
            plt.axis('off')
            plt.imshow(out["decoded"][0].permute(1,2,0).cpu().detach().numpy())
            plt.show()
        #discrete = torch.reshape(discrete, (discrete.shape[0] * discrete.shape[1] * discrete.shape[2], -1))
        # Effective bits ([1] 6.1)
        discrete_usage += discrete.sum(dim=(0)).cpu().detach()
        del discrete

        mse_losses.append(mse(img, out["decoded"]).cpu().detach().numpy())
        bit_variances.append(bit_variance.mean().cpu().detach().numpy())


    return {
        "bit_variance": np.array(bit_variances).mean(),
        "discrete_usage": discrete_usage,
        "reconstruction_error": np.array(mse_losses).mean()
    }

def evaluate_sae(model: StateAE, samples: int = 100):
    return evaluate(model=model, samples=samples, adapt_discrete=lambda out: out["discrete"][:, ::2], batch_size=100)


def evaluate_dsa_combined(model: StateAE, samples: int = 100):
    def adapt_fn(out):
        discrete = out[5][:, ::2]
        return discrete
    return evaluate(model=model, samples=samples, adapt_discrete=adapt_fn)

def evaluate_dsa_per_slot(model: StateAE, samples: int = 100):
    # writer = SummaryWriter(f"test/{parameters.name}", purge_step=0)
    def adapt_fn(out, i=101):
        discrete = torch.reshape(out[5], (-1, parameters.slots, int(2 * parameters.latent_size / parameters.slots)))
        discrete = discrete[:, :, ::2]
        # pivot_state = discrete[0].unsqueeze(0).expand(discrete.shape[0], -1, -1)
        # write_discrete(writer, i, pivot_state, "pivot")
        # write_discrete(writer, i, discrete, "")
        # ordered_states = hungarian_matching(pivot_state, discrete)
        # write_discrete(writer, i, ordered_states)
        return discrete # torch.reshape(ordered_states, (ordered_states.shape[0], -1))
    return evaluate(model=model, samples=samples, adapt_discrete=adapt_fn, latent_space_size=24)

