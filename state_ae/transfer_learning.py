import torch
from torchsummary import summary

from slot_attention_state_ae import DiscreteSlotAttention_model
from parameters import parameters
from train_slot_attention import train

def transfer_learn():
    model = DiscreteSlotAttention_model(parameters.slots, parameters.slot_iters, 0, encoder_hidden_channels=parameters.encoder_hidden_channels,
            attention_hidden_channels=parameters.attention_hidden_channels,
            decoder_hidden_channels=parameters.decoder_hidden_channels,)

    # Compatability for loading weights (differing shapes cause unresolvable error)
    model.mlp_to_gs = None #torch.nn.Sequential(
#        torch.nn.Linear(parameters.encoder_hidden_channels, parameters.fc_width),
 #       torch.nn.Tanh(),
  #      torch.nn.Linear(parameters.fc_width, 2*8),
   # ).to("cuda")

    model.mlp_from_gs = None # torch.nn.Sequential(
#        torch.nn.Linear(2*8, parameters.fc_width),
 #       torch.nn.Tanh(),
  #      torch.nn.Linear(parameters.fc_width, parameters.encoder_hidden_channels)
   # ).to("cuda")

    model = torch.nn.DataParallel(model, device_ids=parameters.device_ids)

    log = torch.load(parameters.resume)
    weights = log["weights"]
    model.load_state_dict(weights, strict=False)
    model.module.discretize = True
    for param in model.parameters():
        param.requires_grad = False
    model.module.encoder_cnn.train()
    model.module.encoder_pos.train()
    model.module.mlp.train()
    model.module.slot_attention.train()
    model.module.decoder_pos.train()
    model.module.decoder_cnn.train()
    #model.module.slot_attention.slots_mu = torch.nn.Parameter(torch.randn(1, 1, parameters.encoder_hidden_channels), requires_grad = False)
    #model.module.slot_attention.slots_log_sigma = torch.nn.Parameter(torch.randn(1, 1, parameters.encoder_hidden_channels), requires_grad = False)

    if parameters.discrete_per_slot:
        raw_size = parameters.encoder_hidden_channels
        discrete_size = int(parameters.latent_size * 2 / parameters.slots)
    else:
        raw_size = parameters.encoder_hidden_channels * parameters.slots
        discrete_size = parameters.latent_size * 2

    model.module.mlp_to_gs = torch.nn.Sequential(
        torch.nn.Linear(raw_size, parameters.fc_width),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(num_features=parameters.fc_width),
        torch.nn.Dropout(p=.4),
        torch.nn.Linear(parameters.fc_width, discrete_size),
    ).to(f"cuda:{parameters.device_ids[0]}")
    model.module.mlp_from_gs = torch.nn.Sequential(
        torch.nn.Linear(discrete_size, parameters.fc_width),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(num_features=parameters.fc_width),
        torch.nn.Dropout(p=.4),
        torch.nn.Linear(parameters.fc_width, raw_size),
    ).to(f"cuda:{parameters.device_ids[0]}")

    learnable_parameters = list(model.module.mlp_to_gs.parameters())
    learnable_parameters += list(model.module.gs.parameters())
    learnable_parameters += list(model.module.mlp_from_gs.parameters())
    learnable_parameters += list(model.module.decoder_pos.parameters())
    learnable_parameters += list(model.module.decoder_cnn.parameters())

    for param in learnable_parameters:
        param.requires_grad = True

#    print(summary(model.module.to(f"cuda:{parameters.device_ids[0]}"), (3, *parameters.image_size)))
    print(f"Using {parameters.resume}")

    train(net=model)

if __name__ == "__main__":
    transfer_learn()
