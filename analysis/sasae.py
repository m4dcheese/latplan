import torch
from state_ae.slot_attention_state_ae import DiscreteSlotAttention_model
from state_ae.parameters import parameters
from state_ae.data import get_loader
import matplotlib.pyplot as plt

model = DiscreteSlotAttention_model(parameters.slots, parameters.slot_iters, 0, discretize=True)

model = torch.nn.DataParallel(model, device_ids=parameters.device_ids)

log = torch.load("/home/madcheese/models/sasae/2022-04-12_22:29:56", map_location="cuda:0")
weights = log["weights"]
model.load_state_dict(weights)

loader = get_loader(dataset="color_shapes", batch_size=1)

i, (x, target) = list(enumerate(loader))[0]

recon_combined, recons, masks, slots, logits, discrete = model(x)
plt.axis('off')
# plt.imshow(x[0].permute((1,2,0)).detach().cpu().numpy())
# fig, axes = plt.subplots(3, 5)
# for i, axis in enumerate(axes.flat):
#     axis.axis('off')
#     axis.imshow(recons[0][i].permute((1,2,0)).detach().cpu().numpy())

# fig, axes = plt.subplots(3, 5)
# for i, axis in enumerate(axes.flat):
#     axis.axis('off')
#     axis.imshow(masks[0][i].permute((1,2,0)).detach().cpu().numpy(), cmap="gray")

fig, axes = plt.subplots(3, 5)
for i, axis in enumerate(axes.flat):
    # axis.axis('off')
    axis.imshow(discrete[i][::2].reshape((4,4)).detach().cpu().numpy(), cmap="Greys")

change_slot = 1
change_index = 4

d = discrete.detach().clone()

d[change_slot][2 * change_index] = 1 - d[change_slot][2 * change_index]
d[change_slot][2 * change_index + 1] = 1 - d[change_slot][2 * change_index + 1]

fig, axes = plt.subplots(3, 5)
for i, axis in enumerate(axes.flat):
    # axis.axis('off')
    axis.imshow(d[i][::2].reshape((4,4)).detach().cpu().numpy(), cmap="Greys")

recon_combined_alt, recons_alt, masks_alt, slots_alt, logits_alt, discrete_alt = model(x, discrete=d)
plt.figure()
plt.axis('off')
plt.imshow(recon_combined[0].permute((1,2,0)).detach().cpu().numpy())

plt.figure()
plt.axis('off')
plt.imshow(recon_combined_alt[0].permute((1,2,0)).detach().cpu().numpy())
plt.tight_layout()
plt.show()