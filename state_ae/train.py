import torch

from data import get_loader
from model import StateAE
from loss import total_loss
from parameters import parameters
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from state_ae.activations import get_tau
from state_ae.util import save_images
from state_ae.utils import save_args


"""
Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""



def train():

    # Store loss terms for graphing
    losses = {
        'z0_prior' : [],
        'x0_recon' : []
    }

    # If usecuda, operations will be performed on GPU
    usecuda = torch.cuda.is_available() and not parameters['no_cuda']
    loader = get_loader(
        dataset="color_shapes",
        blur=parameters.blur,
        deletions=parameters.deletions,
        total_samples=parameters["total_samples"],
        batch_size=parameters["batch_size"],
        field_random_offset=parameters.field_random_offset,
        usecuda=usecuda
    )

    device = torch.device("cuda") if usecuda else torch.device("cpu")
    print("Using device", device)

    # Need to put this tensor on the device, done here instead of passing device
    # TODO why does she do this?
    p = torch.Tensor([parameters['p']]).to(device)

    # Create model
    model = StateAE(parameters, device=device).to(device)
    print(summary(model, (3, 84, 84)))
    optimizer = torch.optim.RAdam(model.parameters(), lr=parameters.lr, weight_decay=1e-5)
    num_steps = len(loader) * parameters.epochs - parameters.warm_up_steps
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00005)

    writer = SummaryWriter(f"runs/{parameters.name}", purge_step=0)
    save_args(parameters, writer)

    model.train()
    torch.set_grad_enabled(True)
    # Training loop
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
            loss, losses = total_loss(out, data[1], p, parameters['beta_kl'], parameters["beta_zs"], epoch=epoch)

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

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("hyper/lr", cur_lr, global_step=i)
            writer.add_scalar("hyper/tau", get_tau(epoch=epoch, total_epochs=parameters.epochs), global_step=i)
            
            if i >= parameters.warm_up_steps:
                scheduler.step()

    return train_loss, losses, model


if __name__=='__main__':
    train_loss, losses, model = train()
