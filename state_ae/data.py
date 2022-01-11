import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
import torchvision.transforms as T

from parameters import parameters

class MNISTPuzzleDataset(Dataset):
    def __init__(self, n: int = 1000, differing_digits: bool = False, deletions: int = 0) -> None:
        super().__init__()
        self.n = n

        # Fetch MNIST digits
        (images, labels), (_, _) = load_data()
        # Group digit images by the actual digit
        digit_classes = [images[labels == i] for i in range(10)]
        digit_classes.append(np.zeros(digit_classes[0].shape).astype(np.uint8))

        # Generate logical puzzle permutations
        permutations = np.array([np.random.permutation(np.arange(9)) for i in range(n)])

        if deletions > 0:
            # Delete some digits
            permutations, permutations_deletion = np.split(permutations, 2, axis=0)

            masks = []
            for dim in permutations_deletion.shape:
                masks.append(np.random.choice(dim, size=deletions))

            permutations_deletion[masks[0], masks[1]] = 10
            permutations = np.append(permutations, permutations_deletion, axis=0)

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
            T.Resize(parameters["image_size"]),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index) -> np.ndarray:
        return self.preprocessing(self.states[index]).float()


def get_loader(
    total_samples: int = 1000,
    batch_size: int = 100,
    differing_digits: bool = False,
    deletions: int = 0,
    usecuda: bool = False
) -> DataLoader:
    """
    Returns a loader for the MNISTPuzzleDataset

    Args:
        total_samples:
            Number of generated samples
        batch_size:
            Batch size for the DataLoader
        differing_digits:
            Whether or not to use different digits for puzzle generation. Default: false
        deletions:
            Total number of removed digits. The deletions only occur in one half of the whole dataset, so that one half of puzzle instances is complete.
        usecuda:
            Whether or not to use cuda
    """
    ds = MNISTPuzzleDataset(n=total_samples, differing_digits=differing_digits, deletions=deletions)
    loader = DataLoader(ds, batch_size=batch_size, pin_memory=usecuda)
    return loader
