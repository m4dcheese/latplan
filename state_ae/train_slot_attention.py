import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
from torch.optim import lr_scheduler
matplotlib.use("Agg")
from torch.utils.tensorboard import SummaryWriter
print(torch.__version__)
from data import get_loader
import slot_attention_state_ae as model
import utils as utils
from slot_attention_obj_discovery.rtpt import RTPT
from parameters import parameters

from util import show_mnist_images

torch.set_num_threads(30)

def run(net, loader, optimizer, criterion, scheduler, writer, args, train=False, epoch=0):
    if train:
        net.train()
        prefix = "train"
        torch.set_grad_enabled(True)
    else:
        net.eval()
        prefix = "test"
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)

    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        imgs = sample.to("cuda:0")
        
        recon_combined, recons, masks, slots = net.forward(imgs)
        loss = criterion(imgs, recon_combined)

        if train:

            if args.resume is None:
                # manual lr warmup
                if i < args.warm_up_steps:
                    learning_rate = args.lr * (i+1)/args.warm_up_steps
                    optimizer.param_groups[0]["lr"] = learning_rate

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 250 == 0:
                utils.write_recon_imgs_plots(writer, i, recon_combined, imgs)
                utils.write_slot_imgs(writer, i, recons)
                utils.write_mask_imgs(writer, i, masks)
                utils.write_slots(writer, i, slots)
                # utils.write_attn(writer, i, net.slot_attention.attn)

                writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
                print(f"Epoch {epoch} Global Step {i} Train Loss: {loss.item():.6f}")

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=i)
            if args.resume is None:
                # normal lr scheduler
                if i >= args.warm_up_steps:
                    scheduler.step()
        else:

            if i % (iters_per_epoch * args.test_log) == 0:
                utils.write_recon_imgs_plots(writer, epoch, recon_combined, imgs, i)

            writer.add_scalar("metric/val_loss", loss.item(), global_step=i)


def train():
    args = parameters
    writer = SummaryWriter(f"runs/{args.name}", purge_step=0)
    # writer = None

    train_loader = get_loader(usecuda=True, batch_size=args.batch_size, total_samples=args.total_samples)

    net = model.SlotAttention_model(n_slots=args.slots, n_iters=args.slot_iters, n_attr=args.slot_attr,
                                    in_channels=1,
                                    encoder_hidden_channels=64, attention_hidden_channels=128,
                                    decoder_hidden_channels=64, decoder_initial_size=(7, 7))

    net = torch.nn.DataParallel(net, device_ids=[0,1,2])
    if args.resume:
        print("Loading ckpt ...")
        log = torch.load(args.resume)
        weights = log["weights"]
        net.load_state_dict(weights, strict=True)


    if not args.no_cuda:
        net = net.to("cuda:0")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_steps = len(train_loader) * args.epochs - args.warm_up_steps
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00005)

    # Create RTPT object
    rtpt = RTPT(name_initials='FM', experiment_name=f"MNIST Puzzle Slot Attn Reconstruction",
                max_iterations=args.epochs)

    # store args as txt file
    utils.save_args(args, writer)

    # Start the RTPT tracking
    rtpt.start()

    for epoch in np.arange(args.epochs):
        run(net, train_loader, optimizer, criterion, scheduler, writer, args,
            train=True, epoch=epoch)
        rtpt.step()

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": vars(args),
        }
        print(os.path.join("logs", args.name))
        torch.save(results, os.path.join("logs", args.name))
        if args.eval_only:
            break
    return net

if __name__ == "__main__":
    net = train()
    torch.save(net.state_dict(), "/home/ml-fmetschies/thesis-latplan/saved")
