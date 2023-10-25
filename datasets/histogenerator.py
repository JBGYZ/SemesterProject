# Description: Generates a dataset of histograms with a given vocabulary
# e.g.: aabbc => [2, 2, 2, 2, 1]; cbba => [1, 2, 2, 1]

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import ortho_group
from torch.utils.data import TensorDataset, DataLoader

def generate_vocabulary(embedding_dim: int = 10, one_hot: bool = False, dummy: bool = False, ):

    if dummy:
        m = torch.zeros(embedding_dim, embedding_dim + 1)
    else:
        m = torch.zeros(embedding_dim, embedding_dim)

    if one_hot:
        torch.diagonal(m, 1)
    else:
        with torch.no_grad():
            m[:, :embedding_dim] = torch.tensor(ortho_group.rvs(dim=embedding_dim), dtype=torch.float32)

    return m

def generate_dataset(vocabulary, data_size):
    dataset = []
    target = []
    for _ in range(data_size):
        sample = np.random.choice(vocabulary.shape[0])
        dataset.append(vocabulary[sample])
        target.append(sample)
    unique, counts = np.unique(target, return_counts=True)
    # replace target values with their frequencies
    target = np.array([counts[unique == t][0] for t in target])
    return torch.from_numpy(np.array(dataset)), torch.from_numpy(np.array(target))


class HistoDataset(Dataset):
    def __init__(
        self,
        data_size: int,
        vocabulary,
        ) :
        super().__init__()
        self.x, self.target = generate_dataset(vocabulary, data_size)   

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """

        x, y = self.x[idx], self.targets[idx]
        return x, y