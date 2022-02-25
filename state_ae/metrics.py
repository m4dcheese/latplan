from typing import Callable
import torch
from tqdm import tqdm

from model import StateAE
from data import get_loader
from parameters import parameters

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
    discrete_usage = torch.zeros(latent_space_size).to(f"cuda:{parameters.device_ids[0]}")
    bit_variances = []
    for i, sample in tqdm(enumerate(loader)):
        # State variance over bits ([1] 6.1)
        discrete_tensors = []
        for iteration in range(100):
            out = model(sample[0].to(f"cuda:{parameters.device_ids[0]}"), epoch=10000)
            discrete = adapt_discrete(out)
            discrete_tensors.append(discrete.cpu().detach())
        discrete_tensors = torch.stack(discrete_tensors)
        bit_variances.append(discrete_tensors.var(dim=0).mean().cpu().detach().numpy())

        # Effective bits ([1] 6.1)
        discrete_usage += discrete.sum(dim=0)
    
    return {
        "bit_variance": sum(bit_variances) / len(bit_variances),
        "discrete_usage": discrete_usage
    }

def evaluate_sae(model: StateAE, samples: int = 100):
    return evaluate(model=model, samples=samples, adapt_discrete=lambda out: out["discrete"], batch_size=100)
    

def evaluate_dsa(model: StateAE, samples: int = 100):
    def adapt_fn(out):
        discrete = out[5][:, :, ::2]
        discrete = discrete.reshape((discrete.shape[0] * parameters.slots, parameters.latent_size))
        return discrete
    return evaluate(model=model, samples=samples, adapt_discrete=adapt_fn, latent_space_size=parameters.latent_size)