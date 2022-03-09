from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from model import StateAE
from data import get_loader
from parameters import parameters
from state_ae.util import hungarian_matching
from torch.utils.tensorboard import SummaryWriter
from utils import write_discrete, write_mask_imgs, write_slot_imgs

def evaluate(
    model: torch.nn.Module,
    samples: int = 100,
    adapt_discrete: Callable = lambda x: x,
    batch_size: int = 10,
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

    # Effective bits - Tensor to save the sum of all propositions
    discrete_usage = torch.zeros(latent_space_size)
    for i, sample in tqdm(enumerate(loader)):
        # State variance over bits ([1] 6.1)
        discrete_tensors = []
        #recons = []
        #masks = []
        if i == 0:
            for iteration in range(1000):
                out = model(sample[0].to(f"cuda:{parameters.device_ids[0]}"), epoch=10000)
                discrete = adapt_discrete(out)
                discrete_tensors.append(discrete.cpu().detach())
                #recons.append(out[1].cpu().detach())
                #masks.append(out[2])
            discrete_tensors = torch.stack(discrete_tensors)
            #recons = torch.stack(recons)
            #masks = torch.stack(masks)
            # for j in range(1, discrete_tensors.shape[0]):
            #     discrete_tensors[j], idx_mapping = hungarian_matching(discrete_tensors[0], discrete_tensors[j])
                #write_discrete(writer, j, discrete_tensors[j])
                #for sample_id in range(1):
                    #idx_mapping_sample = dict(idx_mapping[sample_id])
                    #ordered_slot_recons = recons[j, sample_id, [idx_mapping_sample[k] for k in range(10)]].unsqueeze(0)
                    #ordered_slot_masks = masks[j, sample_id, [idx_mapping_sample[k] for k in range(10)]].unsqueeze(0)
                    #write_slot_imgs(writer, j, ordered_slot_recons.permute(0,1,3,4,2))
                    #print(ordered_slot_masks.shape)
                    #write_mask_imgs(writer, j, ordered_slot_masks.permute(0,1,3,4,2))
            # variance_over_iterations = discrete_tensors.var(dim=0).cpu().detach().numpy()
            # variance_over_iterations_per_slot = variance_over_iterations.mean(axis=2)
            # variance_over_iterations_per_slot[:, variance_over_iterations_per_slot.argmax(axis=1)] = 0
            # bit_variance = variance_over_iterations_per_slot.mean()
            bit_variance = discrete_tensors.var(dim=0).mean().cpu().detach().numpy()
        else:
            out = model(sample[0].to(f"cuda:{parameters.device_ids[0]}"), epoch=10000)
            discrete = adapt_discrete(out)
        discrete = torch.reshape(discrete, (discrete.shape[0], -1))
        # Effective bits ([1] 6.1)
        discrete_usage += discrete.sum(dim=0).cpu().detach()
        del discrete
    
    return {
        "bit_variance": bit_variance,
        "discrete_usage": discrete_usage
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
    return evaluate(model=model, samples=samples, adapt_discrete=adapt_fn)