from torch.utils.data import Dataset, DataLoader
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

class MNISTPuzzleDataset(Dataset):

    def __init__(self, n: int = 1000, differing_digits: bool = False) -> None:
        super().__init__()
        self.n = n
        (images, labels), (_, _) = load_data()

        # Generate logical puzzle permutations
        permutations = np.array([np.random.permutation(np.arange(9)) for i in range(n)])
        index = np.random.randint(0, 10)
        # Split digits into groups of equals
        digit_classes = [images[labels == i] for i in range(10)]

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
        self.states = states
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index) -> np.ndarray:
        return self.states[index]


def get_loader(
    total_samples: int = 1000,
    batch_size: int = 100,
    differing_digits: bool = False,
    usecuda: bool = False
) -> DataLoader:
    ds = MNISTPuzzleDataset(n=total_samples, differing_digits=differing_digits)
    loader = DataLoader(ds, batch_size=batch_size, pin_memory=usecuda)
    return loader