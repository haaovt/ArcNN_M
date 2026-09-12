import torch
import numpy as np
from torch.utils.data import Dataset
import os
import bisect

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, file_loader=None, transform=None, cfgs=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.x_paths = sorted([
            os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir) 
            if p.lower().endswith('_x.npy')
        ])
        self.y_paths = [p.replace('_x.npy', '_y.npy') for p in self.x_paths]
        
        self.cumulative_sizes = []
        total_chunks = 0
        
        for p in self.x_paths:
            # Chỉ đọc header để lấy kích thước (0MB RAM)
            arr_view = np.load(p, mmap_mode='r')
            total_chunks += arr_view.shape[0]
            self.cumulative_sizes.append(total_chunks)
            
        self.total_samples = total_chunks

    def __getitem__(self, index):
        file_idx = bisect.bisect_right(self.cumulative_sizes, index)
        
        if file_idx == 0:
            rel_idx = index
        else:
            rel_idx = index - self.cumulative_sizes[file_idx - 1]
            
        x_mmap = np.load(self.x_paths[file_idx], mmap_mode='r')
        y_mmap = np.load(self.y_paths[file_idx], mmap_mode='r')
        
        x = np.array(x_mmap[rel_idx])
        y = np.array(y_mmap[rel_idx])
        
        if self.transform is not None:
            x = self.transform(x)
            
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.total_samples

class PCADataset(Dataset):
    def __init__(self, feature_path, labels_path):
        super().__init__()
        self.features = np.load(feature_path)
        self.labels = np.load(labels_path)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.labels)