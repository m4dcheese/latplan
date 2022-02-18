import torch
from torchsummary import summary

from slot_attention_state_ae import DiscreteSlotAttention_model
from parameters import parameters
from train_slot_attention import train

def transfer_learn():
    model = DiscreteSlotAttention_model(parameters.slots, parameters.slot_iters, 0)

    model = torch.nn.DataParallel(model)

    log = torch.load(parameters.resume)
    weights = log["weights"]
    model.load_state_dict(weights, strict=False)
    model.module.discretize = True
    for param in model.parameters():
        param.requires_grad = False
    model.module.encoder_cnn.eval()
    model.module.encoder_pos.eval()
    model.module.mlp.eval()
    model.module.slot_attention.eval()
    model.module.decoder_pos.train()
    model.module.decoder_cnn.train()

    model.module.mlp_to_gs = torch.nn.Sequential(
        torch.nn.Linear(parameters.encoder_hidden_channels, parameters.fc_width),
        torch.nn.Tanh(),
        torch.nn.Linear(parameters.fc_width, 2*parameters.latent_size),
    ).to("cuda")

    model.module.mlp_from_gs = torch.nn.Sequential(
        torch.nn.Linear(2*parameters.latent_size, parameters.fc_width),
        torch.nn.Tanh(),
        torch.nn.Linear(parameters.fc_width, parameters.encoder_hidden_channels)
    ).to("cuda")

    learnable_parameters = list(model.module.mlp_to_gs.parameters())
    learnable_parameters += list(model.module.gs.parameters())
    learnable_parameters += list(model.module.mlp_from_gs.parameters())

    for param in learnable_parameters:
        param.requires_grad = True

    # print(summary(model.module, (3, *parameters.image_size)))
    print(f"Using {parameters.resume}")

    train(net=model)

if __name__ == "__main__":
    transfer_learn()