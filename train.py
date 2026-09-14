"""
train.py
--------
Pipeline chính — Train/Val/Test MỘT LẦN theo Bảng II của paper.
KHÔNG còn K-fold.

Raw CSV (đã preprocess thành train/val/test)
    -> ArcNN
    -> feature 1024-D
    -> PCA (fit TRAIN only, EVR=80%)
    -> SHLNN
    -> test cuối cùng trên test set

QUAN TRỌNG VỀ STORAGE:
- Raw data: /kaggle/input (read-only)
- Intermediate feature/PCA: /kaggle/tmp/arcnn
- Final kết quả nhỏ: /kaggle/working/arcnn_output

Không lưu toàn bộ dataset trung gian vào /kaggle/working.
"""

import gc
import json
import os

import numpy as np
import torch

from utils.cmd_parser import get_agrs_parser
from utils.init_utils import get_dataloader
from utils.trainer_utils import BaseTrainer
from utils.pca_utils import run_pca_compression

# ============================================================
# Storage
# ============================================================
SCRATCH_ROOT = os.environ.get(
    "ARCCN_SCRATCH",
    "/kaggle/tmp/arcnn",
)

# Chỉ lưu artifact nhỏ ở working.
FINAL_ROOT = os.environ.get(
    "ARCCN_FINAL",
    "/kaggle/working/arcnn_output",
)

# ============================================================
# Paper hyperparameters
# ============================================================
SEQ_LEN = 512

# Paper: PCA giữ 80% explained variance.
PCA_EVR = 0.80

# Paper: ArcNN learning rate = 0.0002.
ARC_CNN_LR = 2e-4

# Paper: SHLNN learning rate = 0.001.
SHLNN_LR = 1e-3

# Paper: batch size = 64.
PAPER_BATCH_SIZE = 64

# Paper: maximum 50 epochs.
PAPER_MAX_EPOCHS = 50

# Paper: "dừng khi validation loss không giảm quá 3 epoch".
PATIENCE = 3


def prepare_runtime(cfgs, args):
    """Thiết lập device và runtime. rootdir/dataset lấy trực tiếp
    từ config.yaml (không còn override sang cấu trúc Fold_X)."""
    args.cuda = (
        not args.no_cuda
        and torch.cuda.is_available()
    )

    os.makedirs(FINAL_ROOT, exist_ok=True)

    # Binary classification theo paper.
    cfgs["num_classes"] = 2
    # Batch size theo paper.
    cfgs["batch_size"] = PAPER_BATCH_SIZE

    return cfgs, args


def create_results_dir(name):
    """Một thư mục kết quả nhỏ cho mỗi stage (arcnn / shlnn)."""
    result_dir = os.path.join(FINAL_ROOT, name)
    os.makedirs(os.path.join(result_dir, "ckpts"), exist_ok=True)
    return result_dir


def main():
    cfgs, args = get_agrs_parser()
    cfgs, args = prepare_runtime(cfgs, args)

    # Seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("ArcNN paper-aligned Train/Val/Test pipeline (không K-fold)")
    print("=" * 70)
    print(f"Scratch  : {SCRATCH_ROOT}")
    print(f"Final    : {FINAL_ROOT}")
    print(f"Batch    : {PAPER_BATCH_SIZE}")
    print(f"Epochs   : {PAPER_MAX_EPOCHS}")
    print(f"Patience : {PATIENCE}")
    print(f"PCA EVR  : {PCA_EVR}")

    # --------------------------------------------------------
    # 1. Một bộ Train/Val/Test duy nhất.
    #    Yêu cầu: rootdir/dataset/{train,val,test}/*_x.npy,*_y.npy
    #    (do preprocess_utils.process_raw_csv_to_train_val_test tạo ra)
    # --------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloader(cfgs, args)

    # ============================================================
    # 2. Train ArcNN
    # ============================================================
    arcnn_result_dir = create_results_dir("arcnn")

    arcnn_cfg = dict(cfgs)
    arcnn_cfg["model"] = "ArcNN"
    arcnn_cfg["learning_rate"] = ARC_CNN_LR

    print("\n" + "=" * 70)
    print("ARCNN")
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
        test_loader=None,  # Paper: test không lộ trong lúc train ArcNN
        results_dir=arcnn_result_dir,
        patience=PATIENCE,
    )

    print(
        f"\nBest ArcNN epoch: "
        f"{None if arcnn_trainer.best_epoch is None else arcnn_trainer.best_epoch + 1}"
    )

    # ============================================================
    # 3. Extract 1024-D features
    # ============================================================
    pca_dir = os.path.join(SCRATCH_ROOT, "pca")
    os.makedirs(pca_dir, exist_ok=True)

    arcnn_trainer.extract_features(train_loader, os.path.join(pca_dir, "train_features.npy"))
    arcnn_trainer.extract_features(val_loader, os.path.join(pca_dir, "val_features.npy"))
    arcnn_trainer.extract_features(test_loader, os.path.join(pca_dir, "test_features.npy"))

    # ============================================================
    # 4. PCA — fit chỉ trên TRAIN, transform VAL/TEST
    # ============================================================
    print("\n" + "=" * 70)
    print("PCA")
    print("=" * 70)

    pca_dim = run_pca_compression(
        data_dir=pca_dir,
        out_dir=pca_dir,
        target_evr=PCA_EVR,
    )

    # ============================================================
    # 5. DataLoader sau PCA
    # ============================================================
    pca_cfgs = dict(cfgs)
    pca_cfgs["dataset"] = "PCADataset"
    pca_cfgs["rootdir"] = pca_dir

    pca_train_loader, pca_val_loader, pca_test_loader = get_dataloader(pca_cfgs, args)

    # ============================================================
    # 6. Train SHLNN
    # ============================================================
    shlnn_result_dir = create_results_dir("shlnn")

    shlnn_cfg = dict(cfgs)
    shlnn_cfg["model"] = "SHLNN"
    shlnn_cfg["learning_rate"] = SHLNN_LR

    print("\n" + "=" * 70)
    print(f"SHLNN ({pca_dim} -> 110 -> 2)")
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

    # ============================================================
    # 7. Ghi summary
    # ============================================================
    summary = {
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
        os.path.join(FINAL_ROOT, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("TRAIN/VAL/TEST HOÀN TẤT")
    print("=" * 70)
    print(f"Final results: {FINAL_ROOT}")


if __name__ == "__main__":
    main()