"""
init_utils.py
-------------
Dựng DataLoader — MỘT bộ Train/Val/Test duy nhất, KHÔNG còn K-fold.

Yêu cầu cấu trúc thư mục (do preprocess_utils.py tạo ra):

    CustomDataset (raw current, trước PCA):
        rootdir/dataset/train/*_x.npy, *_y.npy
        rootdir/dataset/val/*_x.npy,   *_y.npy
        rootdir/dataset/test/*_x.npy,  *_y.npy

    PCADataset (sau khi extract feature + PCA):
        rootdir/train_features_pca.npy, train_labels.npy
        rootdir/val_features_pca.npy,   val_labels.npy
        rootdir/test_features_pca.npy,  test_labels.npy
"""

import os
from torch.utils.data import DataLoader
from utils.trainer_utils import BaseTrainer
from utils.dataset_utils import CustomDataset, PCADataset

trainers_dict = {'BaseTrainer': BaseTrainer}
datasets_dict = {'CustomDataset': CustomDataset, 'MyData': CustomDataset, 'PCADataset': PCADataset}


def get_trainer(cfgs, args):
    return trainers_dict[cfgs['trainer']](cfgs, args)


def get_dataloader(cfgs, args):
    """
    Trả về (train_loader, val_loader, test_loader) — một bộ duy nhất.
    """
    if cfgs['dataset'] == 'PCADataset':
        return _get_pca_dataloader(cfgs, args)
    return _get_raw_dataloader(cfgs, args)


def _get_pca_dataloader(cfgs, args):
    base_dir = cfgs['rootdir']

    train_dataset = PCADataset(
        os.path.join(base_dir, 'train_features_pca.npy'),
        os.path.join(base_dir, 'train_labels.npy'),
    )
    val_dataset = PCADataset(
        os.path.join(base_dir, 'val_features_pca.npy'),
        os.path.join(base_dir, 'val_labels.npy'),
    )
    test_dataset = PCADataset(
        os.path.join(base_dir, 'test_features_pca.npy'),
        os.path.join(base_dir, 'test_labels.npy'),
    )

    return _build_loaders(train_dataset, val_dataset, test_dataset, cfgs, args)


def _get_raw_dataloader(cfgs, args):
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])

    for split in ('train', 'val', 'test'):
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục '{split}' trong {dataset_dir}. "
                f"Hãy chạy preprocess_utils.process_raw_csv_to_train_val_test "
                f"trước khi train."
            )

    dataset_cls = datasets_dict[cfgs['dataset']] if cfgs['dataset'] in datasets_dict else CustomDataset

    train_dataset = dataset_cls(os.path.join(dataset_dir, 'train'), cfgs=cfgs)
    val_dataset = dataset_cls(os.path.join(dataset_dir, 'val'), cfgs=cfgs)
    test_dataset = dataset_cls(os.path.join(dataset_dir, 'test'), cfgs=cfgs)

    return _build_loaders(train_dataset, val_dataset, test_dataset, cfgs, args)


def _build_loaders(train_dataset, val_dataset, test_dataset, cfgs, args):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfgs['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfgs['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfgs['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader