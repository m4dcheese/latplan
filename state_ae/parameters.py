from datetime import datetime


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parameters = dotdict({
    # General
    "name": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    "epochs": 200,
    "batch_size": 100,
    "no_cuda": False,
    "image_size": (64, 64),
    "warm_up_steps": .2, # Float for percentage of epochs
    "lr": 5e-4,
    "device_ids": [0],
    "deterministic": 1,
    "gaussian_noise": 0.01,

    # Puzzle data
    "total_samples": 20000,
    "deletions": 0,
    "differing_digits": False,
    "blur": .8,
    "remove_target_offset": False,
    "field_random_offset": 0,
    "field_resolution": 24,
    "field_padding": .2, # Float for percentage of field_resolution
    "random_distribution_prob": 0,
    "color_table_size": 3,
    "background": 64,
    "random_colors": False,
    "random_orientation": False,

    # Discretization
    "latent_size": 100,
    "p": 0.1,
    "beta": 0.05,
    "zero_supp_version": "asymmetric",
    "loss_beta_plan": "increase",
    "loss_kwargs": {"fraction_increase_end": 1/3},

    # StateAE architecture
    "fc_width": 1000,
    "dropout": .4,
    "encoder_channels": 16,

    # Slot Attention architecture
    "slots": 10,
    "slot_iters": 3,
    "encoder_hidden_channels": 64,
    "attention_hidden_channels": 128,
    "decoder_hidden_channels": 64
})

if type(parameters.warm_up_steps) == float:
    parameters.warm_up_steps = int(parameters.epochs * parameters.warm_up_steps * parameters.total_samples / parameters.batch_size)