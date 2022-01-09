from datetime import datetime


class ClassFromDict():
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


parameters = ClassFromDict({
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
    "slot_attr": 18,
    "mean_similarity_loss_factor": .2
})
