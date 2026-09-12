import os
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.trainer_utils import BaseTrainer
from utils.dataset_utils import CustomDataset, PCADataset

trainers_dict = {'BaseTrainer': BaseTrainer}
datasets_dict = {'CustomDataset': CustomDataset, 'MyData': CustomDataset, 'PCADataset': PCADataset}

def get_trainer(cfgs, args):
    return trainers_dict[cfgs['trainer']](cfgs, args)

def get_dataloader(cfgs, args):
    if cfgs['dataset'] == 'PCADataset':
        return get_pca_dataloader_folds(cfgs, args)
    return get_dataloader_folds(cfgs, args)

def get_pca_dataloader_folds(cfgs, args):
    loaders = []
    dataset_dir = cfgs['rootdir'] 
    fold_list = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    for fold_dir in fold_list:
        comp_dirs = [d for d in os.listdir(fold_dir) if d.startswith('compressed')]
        if not comp_dirs: continue
        comp_dir = os.path.join(fold_dir, comp_dirs[0])
        train_dataset = PCADataset(os.path.join(comp_dir, 'train_features_pca.npy'), os.path.join(comp_dir, 'train_labels.npy'))
        val_dataset = PCADataset(os.path.join(comp_dir, 'val_features_pca.npy'), os.path.join(comp_dir, 'val_labels.npy'))
        test_dataset = PCADataset(os.path.join(comp_dir, 'test_features_pca.npy'), os.path.join(comp_dir, 'test_labels.npy'))
        train_loaders = DataLoader(dataset=train_dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
        val_loaders = DataLoader(dataset=val_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
        test_loaders = DataLoader(dataset=test_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
        loaders.append((train_loaders, val_loaders, test_loaders))
    return loaders

def get_dataloader_folds(cfgs, args):
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
    subfolders = [os.path.join(dataset_dir, fold) for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, fold))]
    fold_list = subfolders if len(subfolders) > 0 else [dataset_dir]
    loaders = []

    test_fold_cfg = cfgs.get('test_fold', 'None')
    if test_fold_cfg != 'None':    
        target_test_fold = os.path.join(dataset_dir, test_fold_cfg)
        if target_test_fold not in fold_list: raise ValueError(f"Test folder {target_test_fold} not found!")     
        
        train_dataset, val_dataset = [], []
        test_loaders = None
    
        for fold in fold_list:
            dataset = datasets_dict[cfgs['dataset']](fold, cfgs=cfgs)
            if fold == target_test_fold:
                test_loaders = DataLoader(dataset=dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
            else:
                idx = np.arange(len(dataset))
                np.random.shuffle(idx)
                train_dataset.append(Subset(dataset, idx[:int(0.8*len(dataset))+1]))
                val_dataset.append(Subset(dataset, idx[int(0.8*len(dataset))+1:]))
        
        train_loaders = DataLoader(dataset=ConcatDataset(train_dataset), batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
        val_loaders = DataLoader(dataset=ConcatDataset(val_dataset), batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
        loaders.append((train_loaders, val_loaders, test_loaders))
    else:
        if len(fold_list) == 1:
            fold = fold_list[0]
            dataset = datasets_dict[cfgs['dataset']](fold, cfgs=cfgs)
            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            train_end, val_end = int(0.7 * len(dataset)), int(0.2 * len(dataset))
            train_dataset = Subset(dataset, idx[:train_end])
            val_dataset = Subset(dataset, idx[train_end:val_end])
            test_dataset = Subset(dataset, idx[val_end:])
            
            train_loaders = DataLoader(dataset=train_dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
            val_loaders = DataLoader(dataset=val_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
            test_loaders = DataLoader(dataset=test_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
            loaders.append((train_loaders, val_loaders, test_loaders))
        else:
            for current_test_fold in fold_list:
                train_dataset, val_dataset = [], []
                test_loaders = None

                for fold in fold_list:
                    dataset = datasets_dict[cfgs['dataset']](fold, cfgs=cfgs)
                    if fold == current_test_fold:
                        test_loaders = DataLoader(dataset=dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
                    else:
                        idx = np.arange(len(dataset))
                        np.random.shuffle(idx)
                        train_dataset.append(Subset(dataset, idx[:int(0.8*len(dataset))+1]))
                        val_dataset.append(Subset(dataset, idx[int(0.8*len(dataset))+1:]))
                
                train_loaders = DataLoader(dataset=ConcatDataset(train_dataset), batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
                val_loaders = DataLoader(dataset=ConcatDataset(val_dataset), batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
                loaders.append((train_loaders, val_loaders, test_loaders))
    return loaders