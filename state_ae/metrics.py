import torch
from tqdm import tqdm

from model import StateAE
from data import get_loader
from parameters import parameters
from util import set_manual_seed

def evaluate_sae(model: StateAE, samples: int = 100, usecuda: bool = True):
    loader = get_loader(
        dataset="color_shapes",
        image_size=parameters.image_size,
        deletions=parameters.deletions,
        total_samples=samples,
        batch_size=parameters.batch_size,
        usecuda=usecuda
    )

    # Effective bits - Tensor to save the sum of all propositions
    discrete_usage = torch.zeros(parameters.latent_size).to(f"cuda:{parameters.device_ids[0]}")
    for i, sample in tqdm(enumerate(loader)):
        # State variance over bits ([1] 6.1)
        discrete_tensors = []
        for iteration in range(100):
            out = model(sample[0].to(f"cuda:{parameters.device_ids[0]}"), epoch=10000)
            discrete_tensors.append(out["discrete"])
        discrete_tensors = torch.stack(discrete_tensors)
        bit_variance = discrete_tensors.var(dim=0).mean()

        # Effective bits ([1] 6.1)
        discrete_usage += out["discrete"][:, ::2].sum(dim=0)
    
    return {
        "bit_variance": bit_variance,
        "discrete_usage": discrete_usage
    }
