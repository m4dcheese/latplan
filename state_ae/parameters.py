from datetime import datetime


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parameters = dotdict({
    "name": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    "gaussian_noise": .1,
    "fc_width": 1000,
    "dropout": .4,
    "latent_size": 20,
    "p": 0.5,
    "total_samples": 20000,
    "epochs": 100,
    "batch_size": 100,
    "beta_z": .8,
    "no_cuda": False,
    "differing_digits": False,
    "image_size": (84, 84),
    "warm_up_steps": 1000,
    "lr": 1e-3,
    "slots": 10,
    "slot_iters": 3,
    "slot_attr": 18
})
