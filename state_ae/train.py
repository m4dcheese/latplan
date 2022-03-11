from matplotlib import pyplot as plt
import numpy as np
import torch
import os

from data import get_loader
from model import StateAE
from loss import total_loss
from parameters import parameters
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from state_ae.model import get_tau
from state_ae.metrics import evaluate_sae
from state_ae.util import save_images
from state_ae.utils import save_args


"""
Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""



def train(parameters=parameters):

    # Store loss terms for graphing
    losses = {
        'z0_prior' : [],
        'x0_recon' : []
    }

    # If usecuda, operations will be performed on GPU
    usecuda = torch.cuda.is_available() and not parameters['no_cuda']
    loader = get_loader(
        dataset="color_shapes",
        image_size=parameters.image_size,
        differing_digits=parameters.differing_digits,
        deletions=parameters.deletions,
        total_samples=parameters.total_samples,
        batch_size=parameters.batch_size,
        usecuda=usecuda
    )

    device = torch.device(f"cuda:{parameters.device_ids[0]}") if usecuda else torch.device("cpu")
    print("Using device", device)

    # Create model
    model = StateAE(device=device).to(device)
    model = torch.nn.DataParallel(model, device_ids=parameters.device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr, weight_decay=1e-5)
    num_steps = len(loader) * parameters.epochs - parameters.warm_up_steps
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00005)

    writer = SummaryWriter(f"runs/{parameters.name}", purge_step=0)
    save_args(parameters, writer)

    model.train()
    torch.set_grad_enabled(True)
    for epoch in range(parameters['epochs']):

        train_loss = 0
        iters_per_epoch = len(loader)

        for i, data in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
            if i < parameters.warm_up_steps:
                learning_rate = parameters.lr * (i+1)/parameters.warm_up_steps
                optimizer.param_groups[0]["lr"] = learning_rate
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)

            out = model(data[0], epoch)
            loss, losses = total_loss(out, data[1], parameters.p, parameters.beta, step=i, writer=writer, params=parameters)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 250 == 0:
                writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
                writer.add_scalar("metric/kl_loss", losses["kl"], global_step=i)
                writer.add_scalar("metric/zs_loss", losses["zs"], global_step=i)
                writer.add_scalar("metric/recon_loss", losses["recon"], global_step=i)
                print(f"Epoch {epoch} Global Step {i} Train Loss: {loss.item():.6f}")
                save_images(out=out, writer=writer, global_step=i)
                with torch.no_grad():
                    model.eval()
                    metrics = evaluate_sae(model=model, samples=1000)
                    writer.add_scalar("metric/bit_variance", metrics["bit_variance"], global_step=i)
                    effective_bits = (metrics["discrete_usage"].cpu().detach().numpy() >= 1.).sum() / parameters.latent_size
                    writer.add_scalar("metric/effective_bits", effective_bits, global_step=i)
                    model.train()

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("hyper/lr", cur_lr, global_step=i)
            writer.add_scalar("hyper/tau", get_tau(epoch=epoch, total_epochs=parameters.epochs), global_step=i)
            
            if i >= parameters.warm_up_steps:
                scheduler.step()
        results = {
            "name": parameters.name,
            "weights": model.state_dict(),
            "parameters": vars(parameters),
        }
        print(os.path.join("logs", parameters.name))
        torch.save(results, os.path.join("logs", parameters.name))

    return train_loss, losses, model


if __name__=='__main__':
    train_loss, losses, model = train()
