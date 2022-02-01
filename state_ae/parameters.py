from datetime import datetime


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parameters = dotdict({
    # General
    "name": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    "epochs": 100,
    "batch_size": 100,
    "no_cuda": False,
    "image_size": (84, 84),
    "warm_up_steps": 2000,
    "lr": 5e-4,
    "device_ids": [0],

    # Puzzle data
    "total_samples": 20000,
    "deletions": 0,
    "differing_digits": False,
    "blur": .8,
    "field_random_offset": 3,

    # Discretization
    "latent_size": 24,
    "p": 0.1,
    "beta_kl": 0.,
    "beta_zs": 0.,

    # StateAE architecture
    "gaussian_noise": .4,
    "fc_width": 1000,
    "dropout": .4,
    "encoder_channels": 16,

    # Slot Attention architecture
    "slots": 10,
    "slot_iters": 3,
    "encoder_hidden_channels": 64,
    "attention_hidden_channels": 128,
    "decoder_hidden_channels": 64,
})
