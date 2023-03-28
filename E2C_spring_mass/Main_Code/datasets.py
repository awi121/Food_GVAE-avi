import os
from os import path
import numpy as np
import json
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import torch

torch.set_default_dtype(torch.float64)

class spring_mass(Dataset):
    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()
    def __len__(self):
        return len(self._data['samples'])
    def __getitem__(self, index):
        return self._processed[index]

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = sample['before']
                after = sample['after']
                control = sample['control']

                processed.append((np.array(before), np.array(control), np.array(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)
class experimental(Dataset):
    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()
    def __len__(self):
        return len(self._data['samples'])
    def __getitem__(self, index):
        return self._processed[index]

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = sample['before']
                after = sample['after']
                control = sample['control']

                processed.append((np.array(before), np.array(control), np.array(after)))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)