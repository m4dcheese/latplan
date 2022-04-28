from state_ae.data import get_loader
import matplotlib.pyplot as plt

loader = get_loader(dataset="mnist", total_samples=2, image_size=(64, 64))

i, img = list(enumerate(loader))[0]

plt.axis('off')
plt.imshow(img[0].permute(1,2,0), cmap="Greys_r")
plt.show()
plt.axis('off')
plt.imshow(img[1].permute(1,2,0), cmap="Greys_r")
plt.show()
plt.axis('off')
plt.imshow(img[2].permute(1,2,0), cmap="Greys_r")
plt.show()