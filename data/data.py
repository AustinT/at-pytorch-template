from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configs = configs

