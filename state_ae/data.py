import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
import torchvision.transforms as T

from data_shapes import generate_shapes

def generate_permutations(n: int, size: int, deletions: int, deleted_index: int) -> np.array:
    # Generate logical puzzle permutations
    permutations = np.array([np.random.permutation(np.arange(size)) for i in range(n)])

    if deletions > 0:
        # Delete some digits
        permutations, permutations_deletion = np.split(permutations, 2, axis=0)

        masks = []
        for dim in permutations_deletion.shape:
            masks.append(np.random.choice(dim, size=deletions))

        # Set deleted fields to unused index
        permutations_deletion[masks[0], masks[1]] = deleted_index
        permutations = np.append(permutations, permutations_deletion, axis=0)
    
    return permutations


class MNISTPuzzleDataset(Dataset):
    def __init__(self, image_size: tuple, total_samples: int = 1000, deletions: int = 0, differing_digits: bool = False) -> None:
        super().__init__()
        self.n = total_samples

        # Fetch MNIST digits
        (images, labels), (_, _) = load_data()
        # Group digit images by the actual digit
        digit_classes = [images[labels == i] for i in range(10)]
        digit_classes.append(np.zeros(digit_classes[0].shape).astype(np.uint8))

        permutations = generate_permutations(n = total_samples, size=9, deletions=deletions, deleted_index=10)

        states = []

        for permutation in permutations:
            glued_panels = np.ndarray(shape=(9,28,28))
            for i in range(9):
                if differing_digits:
                    index = np.random.randint(0, 10)
                else:
                    index = 0
                glued_panels[i] = digit_classes[permutation[i]][index]
            glued_panels = glued_panels.reshape((3, 3, 28, 28))
            glued_panels = np.concatenate(glued_panels, axis=1)
            glued_panels = np.concatenate(glued_panels, axis=1)
            states.append(glued_panels)

        states = np.array(states)
        states = states.reshape((*states.shape, 1))
        self.states = states.astype(np.float32)
        self.preprocessing = T.Compose([
            T.ToPILImage(), 
            T.Resize(image_size),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index) -> np.ndarray:
        return self.preprocessing(self.states[index]).float()


class ColorShapesPuzzleDataset(Dataset):
    def __init__(self, image_size: tuple, total_samples: int = 1000, deletions: int = 0, blur: float = 0.) -> None:
        super().__init__()
        self.n = total_samples

        # Generate logical puzzle permutations
        permutations = generate_permutations(n=total_samples, size=9, deletions=deletions, deleted_index=9)

        states = generate_shapes(permutations=permutations, blur=blur)

        states = np.array(states)
        self.states = states.astype(np.uint8)
        self.preprocessing = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index) -> np.ndarray:
        return self.preprocessing(self.states[index]).float()


def get_loader(
    dataset: str = "mnist",
    total_samples: int = 1000,
    batch_size: int = 100,
    image_size: tuple = (84, 84),
    deletions: int = 0,
    usecuda: bool = False,
    differing_digits: bool = False,
    blur: float = 0.
) -> DataLoader:
    """
    Returns a loader for the MNISTPuzzleDataset

    Args:
        dataset:
            Specifies the puzzle type. Available: "mnist", "color_shapes"
        total_samples:
            Number of generated samples
        batch_size:
            Batch size for the DataLoader
        image_size:
            Pixel dimensions of sample images
        deletions:
            Total number of removed digits. The deletions only occur in one half of the whole dataset, so that one half of puzzle instances is complete.
        usecuda:
            Whether or not to use cuda
        differing_digits:
            [mnist] Whether or not to use different digits for puzzle generation. Default: false
        blur:
            [color_shapes] Amount of blur (5x5 Gaussian Blur) to apply to shape puzzles
    """
    if dataset == "mnist":
        ds = MNISTPuzzleDataset(
            image_size=image_size,
            total_samples=total_samples,
            deletions=deletions,
            differing_digits=differing_digits
        )
    elif dataset == "color_shapes":
        ds = ColorShapesPuzzleDataset(
            image_size=image_size,
            total_samples=total_samples,
            deletions=deletions,
            blur=blur
        )
    loader = DataLoader(ds, batch_size=batch_size, pin_memory=usecuda, shuffle=True)
    return loader
