import torch
from torch.utils.data import Dataset
import polars as pl
from typing import List, Union


class CustomDataset(Dataset):
    def __init__(self, data: pl.DataFrame, exclude_columns: List[str],  label_column: Union[str, List[str]]= 'tumor_class', dtype=torch.long ,transform=None):
        self.data = data.select(pl.exclude(exclude_columns))
        self.label_column=label_column
        self.labels = data.select(label_column)
        self.transform = transform
        self.dtype=dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.row(idx), dtype=torch.float32)
        if isinstance(self.label_column, str):
            dtype = self.dtype
            label = torch.tensor(self.labels.row(idx)[0], dtype=dtype)
        else:
            label_values = self.labels.row(idx)
            label = {
            col: torch.tensor(val, dtype=self.dtype.get(col, torch.float32))
            for col, val in zip(self.labels.columns, label_values)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample, label

