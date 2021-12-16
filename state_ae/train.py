import torch
import torch.nn as nn
import numpy as np
import itertools

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

    # Run wandb (logs training)
    # if not args['no_wandb']:
    #     import wandb
    #     run = wandb.init(project='latplan-pytorch',
    #         group="%s" % (args['dataset']),
    #         config={'dataset':args['dataset']},
    #         reinit=True)
    # else:
    #     wandb = None
    
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
        print("epoch: {}, train loss = {:.6f}, ".format(epoch + 1, train_loss), end="")

        # Validation - not used to impact model training, essentially working as a testing dataset
        # with torch.no_grad():

        #     model.eval()
        #     val_loss = 0

        #     for data in loaders['val']:

        #         data = data.to(device)
        #         out = model(data, epoch)
        #         loss, _ = total_loss(out, p, args['beta_z'], args['beta_d'])
        #         val_loss += loss.item()
            
        #     val_loss /= (len(loaders['val']) * args['batch'])
        #     print("val loss = {:.6f}".format(val_loss))
        
        #     model.train()
        
        # Save results as gif
    #     if (epoch + 1) % args['save_every'] == 0:
            
    #         pres_dec = out['x_dec_0'].to('cpu').numpy()
    #         sucs_dec = out['x_dec_1'].to('cpu').numpy()
    #         pres_aae = out['x_aae_3'].to('cpu').numpy()
    #         sucs_aae = out['x_aae_2'].to('cpu').numpy()

    #         dec_joint = np.concatenate((pres_dec, sucs_dec), axis=3)
    #         aae_joint = np.concatenate((pres_aae, sucs_aae), axis=3)
    #         joint = np.concatenate((dec_joint, aae_joint), axis=2)
            
    #         save_as_gif(joint, 'saved_gifs/' + str(args['beta_d']) + '_' + str(args['beta_z']) + '_' + str(args['fluents']) + '_' + str(epoch + 1) + '.gif')


    #     # Log results in wandb
    #     if wandb is not None:
    #         wandb.log({"train-loss": train_loss, "val-loss": val_loss})

    # if wandb is not None:
    #     run.finish()
    
    return train_loss, val_loss, losses, parameters



if __name__=='__main__':

    train_loss, val_loss, losses, parameters = train()
    # save_loss_plots(losses, args['beta_d'], args['beta_z'], args['fluents'])
    # print("Beta_d = %d, beta_z = %d, fluents = %d finished with train loss %.5f and val loss %.5f" % (args['beta_d'], args['beta_z'], args['fluents'], train_loss, val_loss))