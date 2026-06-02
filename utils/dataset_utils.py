import torch
import numpy as np
from torch.utils.data import Dataset
import os
import h5py

class CustomDataset(Dataset):
    """
    Custom Dataset class for datasets other than images, look for ImageFolder of torchvision if image dataset is wanted
    """
    def __init__(self, dataset_dir, file_loader, transform=None, cfgs=None):
        """
        Read all samples' path in a dataset folder

        Args:
            dataset_dir (str)   :   path to dataset folder, can use type Path from pathlib if preferred
            file_loader (func)  :   function to read file
            transform   (any)   :   any type, transformations to apply to data
            cfgs         (dict) :   contains all the configurations
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sample_paths = [path for path in os.listdir(dataset_dir) if not os.path.isdir(path)] # can add more conditions to read paths correctly, i.e. if ... and ".h5" in path
        self.file_loader = file_loader
        self.transform = transform

    def __getitem__(self, index):
        """
        Return sample by index using lazy loader

        Args:
            index (int) :   Index

        Return:
            (tuple)   : (sample, target)
        """

        path = self.sample_paths[index]
        sample, target = self.file_loader(os.path.join(self.dataset_dir,path))
        if self.transform is not None:
            sample = self.transform(sample)

        # implement target = self.target_transform(target) if needed

        return sample, target

    def __len__(self):
        return len(self.sample_paths)
    

def CustomFileLoader(path):
    """
    A function to read data of a sample from path. This can be intergrated into CustomDataset.__getitem__() for code simplicity. Changed based on how data is store, here is an example of .h5 file.
    Args:
        path (str)  :   file path to read from
    
    Return:
        (tuple)   : (sample, target)
    """
    if not os.path.isfile(path):
        raise ValueError(f"File {path} cannot be found!")
    with h5py.File(path, 'r') as hf:
        sample = hf['sample'][()]
        target = hf['target'][()]
    return sample, target











