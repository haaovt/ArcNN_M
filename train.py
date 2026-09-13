"""
train.py
--------
Pipeline chính:

K-fold raw CSV
    -> ArcNN
    -> feature 1024-D
    -> PCA (fit TRAIN only, EVR=80%)
    -> SHLNN
    -> final test

Giữ nguyên 5-fold theo yêu cầu của project.

QUAN TRỌNG VỀ STORAGE:
- Raw data: /kaggle/input (read-only)
- Intermediate K-fold/PCA: /kaggle/tmp/arcnn
- Final kết quả nhỏ: /kaggle/working/arcnn_output

Không lưu toàn bộ dataset trung gian vào /kaggle/working.
"""

import gc
import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.cmd_parser import get_agrs_parser
from utils.init_utils import get_dataloader
from utils.dataset_utils import PCADataset
from utils.trainer_utils import BaseTrainer
from utils.pca_utils import run_pca_compression


# ============================================================
# Storage
# ============================================================

# Scratch disk: dùng cho dataset đã preprocess + PCA.
SCRATCH_ROOT = os.environ.get(
    "ARCCN_SCRATCH",
    "/kaggle/tmp/arcnn",
)

SCRATCH_DATASET = os.path.join(
    SCRATCH_ROOT,
    "dataset",
    "MyData",
)

# Chỉ lưu artifact nhỏ ở working.
FINAL_ROOT = os.environ.get(
    "ARCCN_FINAL",
    "/kaggle/working/arcnn_output",
)


# ============================================================
# Paper hyperparameters
# ============================================================

N_FOLDS = 5
SEQ_LEN = 512

# Paper: PCA giữ 80% explained variance.
PCA_EVR = 0.80

# Paper: ArcNN learning rate = 0.0002.
AR_CNN_LR = 2e-4

# Paper: SHLNN learning rate = 0.001.
SHLNN_LR = 1e-3

# Paper: batch size = 64.
PAPER_BATCH_SIZE = 64

# Paper: maximum 50 epochs.
PAPER_MAX_EPOCHS = 50

# Early stopping sau >3 epoch không cải thiện.
PATIENCE = 3


def prepare_runtime(cfgs, args):
    """Thiết lập device và đường dẫn runtime."""
    args.cuda = (
        not args.no_cuda
        and torch.cuda.is_available()
    )

    os.makedirs(
        SCRATCH_DATASET,
        exist_ok=True,
    )
    os.makedirs(
        FINAL_ROOT,
        exist_ok=True,
    )

    # Override rootdir để DataLoader đọc từ scratch.
    cfgs["rootdir"] = os.path.join(
        SCRATCH_ROOT,
        "dataset",
    )
    cfgs["dataset"] = "MyData"

    # Binary classification theo paper.
    cfgs["num_classes"] = 2

    # Batch size theo paper.
    cfgs["batch_size"] = PAPER_BATCH_SIZE

    return cfgs, args


def create_results_dir(fold_idx):
    """
    Mỗi fold chỉ có một thư mục kết quả nhỏ.

    Không lưu raw/preprocessed dataset tại đây.
    """
    fold_dir = os.path.join(
        FINAL_ROOT,
        f"fold_{fold_idx}",
    )

    os.makedirs(
        os.path.join(fold_dir, "arcnn", "ckpts"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(fold_dir, "shlnn", "ckpts"),
        exist_ok=True,
    )

    return fold_dir


def build_pca_dataloaders(pca_dir, batch_size, num_workers):
    """
    Tạo DataLoader cho 3 tập PCA:
        train / val / test
    """
    compressed_dirs = [
        d for d in os.listdir(pca_dir)
        if d.startswith("compressed_")
    ]

    if not compressed_dirs:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục compressed_* trong {pca_dir}"
        )

    # Chỉ có một compressed_* cho mỗi fold.
    compressed_dir = os.path.join(
        pca_dir,
        sorted(compressed_dirs)[0],
    )

    train_dataset = PCADataset(
        os.path.join(
            compressed_dir,
            "train_features_pca.npy",
        ),
        os.path.join(
            compressed_dir,
            "train_labels.npy",
        ),
    )

    val_dataset = PCADataset(
        os.path.join(
            compressed_dir,
            "val_features_pca.npy",
        ),
        os.path.join(
            compressed_dir,
            "val_labels.npy",
        ),
    )

    test_dataset = PCADataset(
        os.path.join(
            compressed_dir,
            "test_features_pca.npy",
        ),
        os.path.join(
            compressed_dir,
            "test_labels.npy",
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
    )


def run_one_fold(
    fold_idx,
    loaders,
    cfgs,
    args,
):
    """
    Chạy trọn pipeline cho một fold.

    loaders:
        train_loader, val_loader, test_loader

    Quy trình:
        1. Train ArcNN.
        2. Restore best ArcNN.
        3. Extract 1024-D features.
        4. PCA fit trên train.
        5. PCA transform val/test.
        6. Train SHLNN.
        7. Restore best SHLNN.
        8. Test cuối cùng.
    """
    train_loader, val_loader, test_loader = loaders

    fold_result_dir = create_results_dir(
        fold_idx
    )

    arcnn_result_dir = os.path.join(
        fold_result_dir,
        "arcnn",
    )

    shlnn_result_dir = os.path.join(
        fold_result_dir,
        "shlnn",
    )

    pca_dir = os.path.join(
        SCRATCH_ROOT,
        "pca",
        f"fold_{fold_idx}",
    )

    os.makedirs(pca_dir, exist_ok=True)

    # ========================================================
    # 1. Train ArcNN
    # ========================================================
    arcnn_cfg = dict(cfgs)

    arcnn_cfg["model"] = "ArcNN"
    arcnn_cfg["learning_rate"] = AR_CNN_LR
    arcnn_cfg["num_epochs"] = PAPER_MAX_EPOCHS
    arcnn_cfg["batch_size"] = PAPER_BATCH_SIZE
    arcnn_cfg["num_classes"] = 2

    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx} - ARCNN")
    print("=" * 70)

    arcnn_trainer = BaseTrainer(
        arcnn_cfg,
        args,
        model_name="ArcNN",
    )

    arcnn_trainer.train(
        num_epochs=PAPER_MAX_EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        results_dir=arcnn_result_dir,
        patience=PATIENCE,
    )

    # ========================================================
    # 2. ArcNN đã tự restore best model.
    # ========================================================
    print(
        f"\nBest ArcNN epoch: "
        f"{None if arcnn_trainer.best_epoch is None else arcnn_trainer.best_epoch + 1}"
    )

    # ========================================================
    # 3. Extract 1024-D features
    # ========================================================
    train_feature_path = os.path.join(
        pca_dir,
        "train_features.npy",
    )

    val_feature_path = os.path.join(
        pca_dir,
        "val_features.npy",
    )

    test_feature_path = os.path.join(
        pca_dir,
        "test_features.npy",
    )

    arcnn_trainer.extract_features(
        train_loader,
        train_feature_path,
    )

    arcnn_trainer.extract_features(
        val_loader,
        val_feature_path,
    )

    arcnn_trainer.extract_features(
        test_loader,
        test_feature_path,
    )

    # ========================================================
    # 4. PCA
    #
    # PCA chỉ fit trên TRAIN.
    # Validation/Test chỉ transform.
    # ========================================================
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx} - PCA")
    print("=" * 70)

    run_pca_compression(
        data_dir=pca_dir,
        target_evr=PCA_EVR,
    )

    compressed_dirs = [
        d for d in os.listdir(pca_dir)
        if d.startswith("compressed_")
    ]

    compressed_dir = os.path.join(
        pca_dir,
        sorted(compressed_dirs)[0],
    )

    # Xác định số chiều PCA thực tế.
    pca_train = np.load(
        os.path.join(
            compressed_dir,
            "train_features_pca.npy",
        ),
        mmap_mode="r",
    )

    pca_dim = int(pca_train.shape[1])

    print(
        f"PCA dimensions = {pca_dim} "
        f"(paper báo cáo 70 dimensions)"
    )

    # ========================================================
    # 5. DataLoader sau PCA
    # ========================================================
    (
        pca_train_loader,
        pca_val_loader,
        pca_test_loader,
        _,
    ) = build_pca_dataloaders(
        pca_dir,
        batch_size=PAPER_BATCH_SIZE,
        num_workers=args.num_workers,
    )

    # ========================================================
    # 6. Train SHLNN
    # ========================================================
    shlnn_cfg = dict(cfgs)

    shlnn_cfg["model"] = "SHLNN"
    shlnn_cfg["learning_rate"] = SHLNN_LR
    shlnn_cfg["num_epochs"] = PAPER_MAX_EPOCHS
    shlnn_cfg["batch_size"] = PAPER_BATCH_SIZE
    shlnn_cfg["num_classes"] = 2

    print("\n" + "=" * 70)
    print(
        f"FOLD {fold_idx} - SHLNN "
        f"({pca_dim} -> 110 -> 2)"
    )
    print("=" * 70)

    shlnn_trainer = BaseTrainer(
        shlnn_cfg,
        args,
        model_name="SHLNN",
        num_inputs=pca_dim,
    )

    shlnn_trainer.train(
        num_epochs=PAPER_MAX_EPOCHS,
        train_loader=pca_train_loader,
        val_loader=pca_val_loader,
        test_loader=pca_test_loader,
        results_dir=shlnn_result_dir,
        patience=PATIENCE,
    )

    # ========================================================
    # 7. Ghi summary của fold
    # ========================================================
    fold_summary = {
        "fold": fold_idx,
        "pca_dimensions": pca_dim,
        "arcnn_best_epoch": (
            None
            if arcnn_trainer.best_epoch is None
            else int(arcnn_trainer.best_epoch + 1)
        ),
        "shlnn_best_epoch": (
            None
            if shlnn_trainer.best_epoch is None
            else int(shlnn_trainer.best_epoch + 1)
        ),
    }

    with open(
        os.path.join(
            fold_result_dir,
            "fold_summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fold_summary,
            f,
            indent=2,
        )

    # ========================================================
    # 8. Giải phóng RAM/VRAM trước fold tiếp theo
    # ========================================================
    del arcnn_trainer
    del shlnn_trainer
    del pca_train
    del (
        pca_train_loader,
        pca_val_loader,
        pca_test_loader,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_summary


def main():
    cfgs, args = get_agrs_parser()

    cfgs, args = prepare_runtime(
        cfgs,
        args,
    )

    # Seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("ArcNN paper-aligned K-Fold pipeline")
    print("=" * 70)
    print(f"Scratch : {SCRATCH_ROOT}")
    print(f"Dataset : {SCRATCH_DATASET}")
    print(f"Final   : {FINAL_ROOT}")
    print(f"Folds   : {N_FOLDS}")
    print(f"Batch   : {PAPER_BATCH_SIZE}")
    print(f"Epochs  : {PAPER_MAX_EPOCHS}")
    print(f"PCA EVR : {PCA_EVR}")

    # --------------------------------------------------------
    # DataLoader vẫn dùng cơ chế K-fold của repo.
    #
    # Nếu có 5 Fold_1 ... Fold_5:
    # mỗi fold lần lượt làm test,
    # 4 fold còn lại được chia train/validation.
    # --------------------------------------------------------
    loaders = get_dataloader(
        cfgs,
        args,
    )

    if len(loaders) != N_FOLDS:
        print(
            f"WARNING: tìm thấy {len(loaders)} fold loader, "
            f"không phải {N_FOLDS}."
        )

    all_summaries = []

    for fold_idx, fold_loaders in enumerate(
        loaders,
        start=1,
    ):
        summary = run_one_fold(
            fold_idx,
            fold_loaders,
            cfgs,
            args,
        )

        all_summaries.append(summary)

        # ----------------------------------------------------
        # Xóa PCA intermediate của fold sau khi SHLNN xong.
        #
        # Điều này giảm disk usage rất mạnh.
        # ArcNN/SHLNN best checkpoint + metrics vẫn nằm ở FINAL_ROOT.
        # ----------------------------------------------------
        fold_pca_dir = os.path.join(
            SCRATCH_ROOT,
            "pca",
            f"fold_{fold_idx}",
        )

        if os.path.isdir(fold_pca_dir):
            shutil.rmtree(
                fold_pca_dir,
                ignore_errors=True,
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Tổng hợp kết quả 5-fold
    # ========================================================
    summary_path = os.path.join(
        FINAL_ROOT,
        "kfold_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_summaries,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("5-FOLD TRAINING COMPLETED")
    print("=" * 70)
    print(f"Final results: {FINAL_ROOT}")


if __name__ == "__main__":
    main()
