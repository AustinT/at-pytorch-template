from data.data import Dataset
import torch
import numpy as np

class SinxDataset(Dataset):

    def __len__(self):
        return 1000

    def __getitem__(self, i):
       x = np.random.uniform(-1, 1, size=(1, 1)).astype(np.float32)
       y = np.sin(x)
       return {'x': torch.from_numpy(x), 'y': torch.from_numpy(y)}

