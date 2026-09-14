"""
init_utils.py
-------------
Xây dựng DataLoader cho ArcNN (raw window) và SHLNN (feature sau PCA).

KHÔNG còn K-fold: dữ liệu đã được preprocess_utils.py chia sẵn thành
3 thư mục cố định Train / Val / Test (xem preprocess_utils.py), và
get_dataloader() ở đây chỉ việc load trực tiếp 3 thư mục đó — không có
vòng lặp qua nhiều fold, không có logic random Subset train/val/test
nữa (nguồn gây bug ở bản cũ).

get_dataloader() trả về TRỰC TIẾP một tuple:
    train_loader, val_loader, test_loader
(không còn là list các fold như trước).
"""

import os
from torch.utils.data import DataLoader
from utils.trainer_utils import BaseTrainer
from utils.dataset_utils import CustomDataset, PCADataset

trainers_dict = {'BaseTrainer': BaseTrainer}
datasets_dict = {'CustomDataset': CustomDataset, 'MyData': CustomDataset}


def get_trainer(cfgs, args):
    return trainers_dict[cfgs['trainer']](cfgs, args)


def get_dataloader(cfgs, args):
    """
    cfgs['dataset'] == 'PCADataset' -> load feature 1024-D đã qua PCA
    (dùng cho SHLNN, qua initialize.py/manual flow).
    Ngược lại -> load raw window 512 điểm (dùng cho ArcNN).
    """
    if cfgs['dataset'] == 'PCADataset':
        return get_pca_dataloader(cfgs, args)
    return get_raw_dataloader(cfgs, args)


def get_raw_dataloader(cfgs, args):
    """
    Load trực tiếp 3 thư mục cố định:
        <rootdir>/<dataset>/Train
        <rootdir>/<dataset>/Val
        <rootdir>/<dataset>/Test

    Các thư mục này do preprocess_utils.py (process_raw_csv_to_split)
    tạo sẵn — chia theo FILE, một lần duy nhất, không K-fold.
    """
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
    train_dir = os.path.join(dataset_dir, 'Train')
    val_dir = os.path.join(dataset_dir, 'Val')
    test_dir = os.path.join(dataset_dir, 'Test')

    for split_name, split_dir in (('Train', train_dir), ('Val', val_dir), ('Test', test_dir)):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục {split_name} tại {split_dir}. "
                f"Hãy chạy preprocess_utils.py (process_raw_csv_to_split) trước."
            )

    train_dataset = datasets_dict[cfgs['dataset']](train_dir, cfgs=cfgs) \
        if cfgs['dataset'] in datasets_dict else CustomDataset(train_dir, cfgs=cfgs)
    val_dataset = CustomDataset(val_dir, cfgs=cfgs)
    test_dataset = CustomDataset(test_dir, cfgs=cfgs)

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


def get_pca_dataloader(cfgs, args):
    """
    Load thư mục PCA đã nén (1 tập duy nhất, do pca_utils.run_pca_compression
    tạo ra): cfgs['rootdir'] phải trỏ thẳng tới thư mục compressed_Xd.
    """
    pca_dir = cfgs['rootdir']

    train_dataset = PCADataset(
        os.path.join(pca_dir, 'train_features_pca.npy'),
        os.path.join(pca_dir, 'train_labels.npy'),
    )
    val_dataset = PCADataset(
        os.path.join(pca_dir, 'val_features_pca.npy'),
        os.path.join(pca_dir, 'val_labels.npy'),
    )
    test_dataset = PCADataset(
        os.path.join(pca_dir, 'test_features_pca.npy'),
        os.path.join(pca_dir, 'test_labels.npy'),
    )

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