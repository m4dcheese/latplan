import matplotlib.pyplot as plt
import numpy as np

def show_mnist_images(data: np.ndarray) -> None:
    num_images = len(data)

    fig, ax = plt.subplots(1, num_images)

    if num_images == 1:
        image = np.array(data[0])
        ax.imshow(image, cmap="Greys")
    else:
        for a, image in zip(ax, data):
            image = np.array(image)
            a.imshow(image, cmap="Greys")
    
    plt.show()