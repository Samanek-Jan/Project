import torch
import datasets
from datasets import IterableDataset, load_dataset

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, shuffle_buffer_size = 1000):
        super(Dataset, self).__init__()
        d : IterableDataset = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["C++"])
        self.len = d.info
        self.ds = iter(d.shuffle(shuffle_buffer_size))

    def __getitem__(self, _ : int):
        return next(self.ds)
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    d = Dataset()
    print(d.__len__())
    print(d.__getitem__(0))