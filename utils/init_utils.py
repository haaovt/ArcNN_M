import os
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.trainer_utils import BaseTrainer
from utils.dataset_utils import CustomDataset, CustomFileLoader

trainers_dict = {
    'BaseTrainer'   :   BaseTrainer
}

datasets_dict = {
    'CustomDataset'   : CustomDataset
}


def get_trainer(cfgs, args):
    return trainers_dict[cfgs['trainer']](cfgs, args)

def get_dataloader(cfgs, args):
    """
    Return loader which is a list of (train_loaders, val_loaders, test_loaders), the list only have 1 element if test_fold is given
    """
    # iterate through all data sub-folders, change if not using sub-folders
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
    fold_list = [os.path.join(dataset_dir,fold) for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,fold))]
    loaders = []

    if cfgs['test_fold'] != 'None':         
        # This code is using train-validation split. Additional modifications is required to handle cfgs['val_fold'] != 'None'.
        dataset = datasets_dict[cfgs['dataset']](fold, file_loader=CustomFileLoader, cfgs=cfgs)
        train_dataset = []
        val_dataset = []
        for fold in fold_list:
            dataset = datasets_dict[cfgs['dataset']](fold, file_loader=CustomFileLoader, cfgs=cfgs)
            if fold != test_fold:
                idx = np.arange(len(dataset))
                np.random.shuffle(idx)
                # Fair 80-20 split between folders
                train_dataset.append(Subset(dataset, idx[:int(0.8*len(dataset))+1]))
                val_dataset.append(Subset(dataset, idx[int(0.8*len(dataset))+1:]))
        
        train_dataset = ConcatDataset(train_dataset)
        val_dataset = ConcatDataset(val_dataset)

        train_loaders = DataLoader(dataset=train_dataset,
                                    batch_size=cfgs['batch_size'],
                                    shuffle=True,
                                    num_workers=args.num_workers
                                    )
        
        val_loaders = DataLoader(dataset=val_dataset,
                                batch_size=cfgs['batch_size'],
                                shuffle=False,
                                num_workers=args.num_workers
                                )
        
        test_dataset = datasets_dict[cfgs['dataset']](cfgs['test_fold'], file_loader=CustomFileLoader, cfgs=cfgs)
        test_loaders = DataLoader(dataset=dataset,
                            batch_size=cfgs['batch_size'],
                            shuffle=False,
                            num_workers=args.num_workers
                            )
        
        loaders.append((train_loaders, val_loaders, test_loaders))
            
    elif cfgs['test_fold'] not in fold_list:
        raise ValueError(f"Test folder {os.path.join(dataset_dir,test_fold)} not found!")
    else:
        # This handles the case where no test is given, each sub-folder takes turn as test folder
        for i_test_fold, test_fold in enumerate(fold_list):
            train_dataset = []
            val_dataset = []

            for fold in fold_list:
                dataset = datasets_dict[cfgs['dataset']](fold, file_loader=CustomFileLoader, cfgs=cfgs)
                if fold == test_fold:
                    test_loaders = DataLoader(dataset=dataset,
                                     batch_size=cfgs['batch_size'],
                                     shuffle=False,
                                     num_workers=args.num_workers
                                     )

                else:
                    idx = np.arange(len(dataset))
                    np.random.shuffle(idx)
                    train_dataset.append(Subset(dataset, idx[:int(0.8*len(dataset))+1]))
                    val_dataset.append(Subset(dataset, idx[int(0.8*len(dataset))+1:]))
            
            train_dataset = ConcatDataset(train_dataset)
            val_dataset = ConcatDataset(val_dataset)

            train_loaders = DataLoader(dataset=train_dataset,
                                        batch_size=cfgs['batch_size'],
                                        shuffle=True,
                                        num_workers=args.num_workers
                                        )
            
            val_loaders = DataLoader(dataset=val_dataset,
                                    batch_size=cfgs['batch_size'],
                                    shuffle=False,
                                    num_workers=args.num_workers
                                    )
            
            loaders.append((train_loaders, val_loaders, test_loaders))
    
    return loaders
       

