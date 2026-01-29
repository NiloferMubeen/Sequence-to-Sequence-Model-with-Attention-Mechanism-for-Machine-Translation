import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        return torch.tensor(src), torch.tensor(trg)


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, padding_value=0)
    trg_padded = pad_sequence(trg_batch, padding_value=0)

    return src_padded, trg_padded
