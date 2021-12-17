import torch

from data import get_loader
from model import StateAE
from loss import total_loss
from parameters import parameters

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
        total_samples=parameters["total_samples"],
        batch_size=parameters["batch_size"],
        differing_digits=parameters["differing_digits"],
        usecuda=usecuda
    )

    device = torch.device("cuda") if usecuda else torch.device("cpu")
    print("Using device", device)

    # Need to put this tensor on the device, done here instead of passing device
    # TODO why does she do this?
    p = torch.Tensor([parameters['p']]).to(device)

    # Create model
    model = StateAE(parameters, device=device).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    for epoch in range(parameters['epochs']):

        train_loss = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, epoch)
            loss, losses = total_loss(out, p, parameters['beta_z'], losses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= (len(loader) * parameters['batch_size'])
        print(f"In epoch: {epoch + 1}, train loss = {train_loss:.6f}, ")

    return train_loss, losses, model


if __name__=='__main__':
    train_loss, losses, model = train()
