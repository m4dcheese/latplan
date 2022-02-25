import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
from torch.optim import lr_scheduler
from state_ae.activations import get_tau
from state_ae.loss import total_loss
matplotlib.use("Agg")
from torch.utils.tensorboard import SummaryWriter
print(torch.__version__)
from data import get_loader
import slot_attention_state_ae as model
import utils as utils
from slot_attention_obj_discovery.rtpt import RTPT
from parameters import parameters
from util import set_manual_seed

torch.set_num_threads(30)

def run(net, loader, optimizer, criterion, scheduler, writer, parameters, epoch=0):
    net.train()
    torch.set_grad_enabled(True)

    iters_per_epoch = len(loader)

    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        imgs = sample[0].to(f"cuda:{parameters.device_ids[0]}"), sample[1].to(f"cuda:{parameters.device_ids[0]}")
        
        if net.module.discretize:
            recon_combined, recons, masks, slots, logits, discrete = net.forward(imgs[0], epoch)
            discrete_batch = torch.reshape(discrete, (discrete.shape[0] * discrete.shape[1], discrete.shape[2]))
            out = {
                "encoded": logits,
                "discrete": discrete_batch,
                "decoded": recon_combined
            }
            loss, losses = total_loss(out, imgs[1], parameters.p, parameters.beta, step=i, writer=writer)
        else:
            recon_combined, recons, masks, slots = net.forward(imgs[0], epoch)

            loss = criterion(imgs[1], recon_combined)

        if parameters.resume is None or True:
            # manual lr warmup
            if i < parameters.warm_up_steps:
                learning_rate = parameters.lr * (i+1)/parameters.warm_up_steps
                optimizer.param_groups[0]["lr"] = learning_rate

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 250 == 0:
            imgs = imgs[0].permute((0,2,3,1))
            recon_combined = recon_combined.permute((0,2,3,1))
            utils.write_recon_imgs_plots(writer, i, recon_combined, imgs)
            recons = recons.permute((0,1,3,4,2))
            utils.write_slot_imgs(writer, i, recons)
            utils.write_mask_imgs(writer, i, masks)
            utils.write_slots(writer, i, slots)
            if net.module.discretize:
                utils.write_discrete(writer, i, discrete)
                writer.add_scalar("metric/zs_loss", losses["zs"].item(), global_step=i)
                writer.add_scalar("metric/recon_loss", losses["recon"].item(), global_step=i)

            writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
            print(f"Epoch {epoch} Global Step {i} Train Loss: {loss.item():.6f}")

        cur_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("hyper/lr", cur_lr, global_step=i)
        writer.add_scalar("hyper/tau", get_tau(epoch, total_epochs=parameters.epochs), global_step=i)
        if parameters.resume is None or True:
            # normal lr scheduler
            if i >= parameters.warm_up_steps:
                scheduler.step()


def train(net: torch.nn.Module = None):
    writer = SummaryWriter(f"runs/{parameters.name}", purge_step=0)
    if parameters.deterministic:
        set_manual_seed(parameters.deterministic)

    train_loader = get_loader(
        dataset="color_shapes",
        usecuda=True,
        image_size=parameters.image_size,
        batch_size=parameters.batch_size,
        total_samples=parameters.total_samples,
        deletions=parameters.deletions,
    )

    if net is None:
        net = model.DiscreteSlotAttention_model(
            n_slots=parameters.slots,
            n_iters=parameters.slot_iters,
            n_attr=parameters.slot_attr,
            in_channels=3,
            encoder_hidden_channels=parameters.encoder_hidden_channels,
            attention_hidden_channels=parameters.attention_hidden_channels,
            decoder_hidden_channels=parameters.decoder_hidden_channels,
            decoder_initial_size=(8, 8),
            discretize=False
        )
        net = torch.nn.DataParallel(net, device_ids=parameters.device_ids)

        if parameters.resume:
            print("Loading ckpt ...")
            log = torch.load(parameters.resume)
            weights = log["weights"]
            net.load_state_dict(weights, strict=False)
            print("Loaded weights from "+parameters.resume)


    if not parameters.no_cuda:
        net = net.to(f"cuda:{parameters.device_ids[0]}")

    optimizer = torch.optim.Adam(net.parameters(), lr=parameters.lr)
    criterion = torch.nn.MSELoss()
    num_steps = len(train_loader) * parameters.epochs - parameters.warm_up_steps
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00005)

    # Create RTPT object
    rtpt = RTPT(name_initials='FM', experiment_name=f"MNIST Puzzle Slot Attn Reconstruction",
                max_iterations=parameters.epochs)

    # store parameters as txt file
    utils.save_args(parameters, writer)

    # Start the RTPT tracking
    rtpt.start()

    for epoch in np.arange(parameters.epochs):
        run(net, train_loader, optimizer, criterion, scheduler, writer, parameters, epoch=epoch)
        rtpt.step()

        results = {
            "name": parameters.name,
            "weights": net.state_dict(),
            "parameters": vars(parameters),
        }
        print(os.path.join("logs", parameters.name))
        torch.save(results, os.path.join("logs", parameters.name))
        if parameters.eval_only:
            break
    return net

if __name__ == "__main__":
    net = train()
    torch.save(net.state_dict(), "/home/ml-fmetschies/thesis-latplan/saved")
