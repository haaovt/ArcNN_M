"""
train.py
--------
Pipeline chính - K-fold cross validation chuẩn:

  5 Fold_1...Fold_5 trên disk
      |
      v
  Vòng 1: test=Fold_1 | train+val = Fold_2+3+4+5 (80/20)
  Vòng 2: test=Fold_2 | train+val = Fold_1+3+4+5 (80/20)
  Vòng 3: test=Fold_3 | train+val = Fold_1+2+4+5 (80/20)
  Vòng 4: test=Fold_4 | train+val = Fold_1+2+3+5 (80/20)
  Vòng 5: test=Fold_5 | train+val = Fold_1+2+3+4 (80/20)
      |
      v
  Mỗi vòng: ArcNN -> PCA (fit train only) -> SHLNN
      |
      v
  Tổng hợp mean accuracy 5 vòng

STORAGE:
- Raw data         : /kaggle/input (read-only)
- Intermediate PCA : /kaggle/tmp/arcnn
- Final results    : /kaggle/working/arcnn_output
"""

import copy
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
SCRATCH_ROOT = os.environ.get("ARCCN_SCRATCH", "/kaggle/tmp/arcnn")
SCRATCH_DATASET = os.path.join(SCRATCH_ROOT, "dataset", "MyData")
FINAL_ROOT = os.environ.get("ARCCN_FINAL", "/kaggle/working/arcnn_output")


# ============================================================
# Paper hyperparameters
# ============================================================
N_FOLDS          = 5
SEQ_LEN          = 512
PCA_EVR          = 0.80    # Paper: 80% explained variance
AR_CNN_LR        = 2e-4    # Paper: ArcNN learning rate
SHLNN_LR         = 1e-3    # Paper: SHLNN learning rate
PAPER_BATCH_SIZE = 64      # Paper: batch size
PAPER_MAX_EPOCHS = 50      # Paper: max epochs
PATIENCE         = 3       # Paper: early stopping patience > 3 epochs


def prepare_runtime(cfgs, args):
    """Thiết lập device và đường dẫn runtime."""
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    os.makedirs(SCRATCH_DATASET, exist_ok=True)
    os.makedirs(FINAL_ROOT, exist_ok=True)

    # DataLoader đọc từ scratch
    cfgs["rootdir"] = os.path.join(SCRATCH_ROOT, "dataset")
    cfgs["dataset"] = "MyData"
    cfgs["num_classes"] = 2
    cfgs["batch_size"] = PAPER_BATCH_SIZE
    return cfgs, args


def create_results_dir(fold_idx):
    """Tạo thư mục kết quả cho mỗi vòng fold."""
    fold_dir = os.path.join(FINAL_ROOT, f"fold_{fold_idx}")
    os.makedirs(os.path.join(fold_dir, "arcnn", "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "shlnn", "ckpts"), exist_ok=True)
    return fold_dir


def build_pca_dataloaders(pca_dir, batch_size, num_workers):
    """Tạo DataLoader cho train/val/test sau PCA."""
    compressed_dirs = [
        d for d in os.listdir(pca_dir)
        if d.startswith("compressed_")
    ]
    if not compressed_dirs:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục compressed_* trong {pca_dir}"
        )
    compressed_dir = os.path.join(pca_dir, sorted(compressed_dirs)[0])

    train_loader = DataLoader(
        PCADataset(
            os.path.join(compressed_dir, "train_features_pca.npy"),
            os.path.join(compressed_dir, "train_labels.npy"),
        ),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        PCADataset(
            os.path.join(compressed_dir, "val_features_pca.npy"),
            os.path.join(compressed_dir, "val_labels.npy"),
        ),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        PCADataset(
            os.path.join(compressed_dir, "test_features_pca.npy"),
            os.path.join(compressed_dir, "test_labels.npy"),
        ),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def run_one_fold(fold_idx, loaders, cfgs, args):
    """
    Chạy pipeline đầy đủ cho một vòng K-fold.

    loaders = (train_loader, val_loader, test_loader)
    Trong đó:
        train_loader = ConcatDataset của 4 fold còn lại, phần 80%
        val_loader   = ConcatDataset của 4 fold còn lại, phần 20%
        test_loader  = fold hiện tại (toàn bộ, không tham gia training)

    Quy trình:
        1. Train ArcNN trên train_loader, chọn best theo val_loader
        2. Extract 1024-D features cho train/val/test
        3. PCA fit chỉ trên train features (EVR=80%)
        4. Transform val/test bằng PCA đã fit
        5. Train SHLNN trên PCA features
        6. Evaluate cuối trên test (fold bị giữ lại)
    """
    train_loader, val_loader, test_loader = loaders

    fold_result_dir  = create_results_dir(fold_idx)
    arcnn_result_dir = os.path.join(fold_result_dir, "arcnn")
    shlnn_result_dir = os.path.join(fold_result_dir, "shlnn")
    pca_dir = os.path.join(SCRATCH_ROOT, "pca", f"fold_{fold_idx}")
    os.makedirs(pca_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Train ArcNN
    #    - train trên 4 fold (80%)
    #    - validate trên 4 fold (20%)
    #    - test_loader=None: test set KHÔNG được nhìn thấy
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx}/5  —  ARCNN TRAINING")
    print(f"  train = 4 folds (80%)  |  val = 4 folds (20%)  |  test = Fold_{fold_idx} (held-out)")
    print("=" * 70)

    arcnn_cfg = copy.deepcopy(cfgs)
    arcnn_cfg["model"]         = "ArcNN"
    arcnn_cfg["learning_rate"] = AR_CNN_LR
    arcnn_cfg["num_epochs"]    = PAPER_MAX_EPOCHS
    arcnn_cfg["batch_size"]    = PAPER_BATCH_SIZE
    arcnn_cfg["num_classes"]   = 2

    arcnn_trainer = BaseTrainer(arcnn_cfg, args, model_name="ArcNN")
    arcnn_trainer.train(
        num_epochs=PAPER_MAX_EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,          # Paper: test không lộ trong training
        results_dir=arcnn_result_dir,
        patience=PATIENCE,
    )

    print(f"\nBest ArcNN epoch: {arcnn_trainer.best_epoch + 1 if arcnn_trainer.best_epoch is not None else 'N/A'}")

    # --------------------------------------------------------
    # 2. Extract 1024-D features
    #    train/val dùng để fit PCA
    #    test  dùng để evaluate cuối
    # --------------------------------------------------------
    train_feat_path = os.path.join(pca_dir, "train_features.npy")
    val_feat_path   = os.path.join(pca_dir, "val_features.npy")
    test_feat_path  = os.path.join(pca_dir, "test_features.npy")

    print(f"\nExtracting features...")
    arcnn_trainer.extract_features(train_loader, train_feat_path)
    arcnn_trainer.extract_features(val_loader,   val_feat_path)
    arcnn_trainer.extract_features(test_loader,  test_feat_path)

    # --------------------------------------------------------
    # 3 & 4. PCA
    #    fit_transform trên train
    #    transform     trên val và test
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx}/5  —  PCA  (EVR={PCA_EVR*100:.0f}%)")
    print("=" * 70)

    run_pca_compression(data_dir=pca_dir, target_evr=PCA_EVR)

    compressed_dirs = [d for d in os.listdir(pca_dir) if d.startswith("compressed_")]
    compressed_dir  = os.path.join(pca_dir, sorted(compressed_dirs)[0])
    pca_dim = int(np.load(
        os.path.join(compressed_dir, "train_features_pca.npy"),
        mmap_mode="r",
    ).shape[1])

    print(f"PCA dimensions = {pca_dim}  (paper báo cáo 70)")

    # --------------------------------------------------------
    # 5. Train SHLNN trên PCA features
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx}/5  —  SHLNN  ({pca_dim} → 110 → 2)")
    print("=" * 70)

    pca_train_loader, pca_val_loader, pca_test_loader = build_pca_dataloaders(
        pca_dir, batch_size=PAPER_BATCH_SIZE, num_workers=args.num_workers,
    )

    shlnn_cfg = copy.deepcopy(cfgs)
    shlnn_cfg["model"]         = "SHLNN"
    shlnn_cfg["learning_rate"] = SHLNN_LR
    shlnn_cfg["num_epochs"]    = PAPER_MAX_EPOCHS
    shlnn_cfg["batch_size"]    = PAPER_BATCH_SIZE
    shlnn_cfg["num_classes"]   = 2

    shlnn_trainer = BaseTrainer(shlnn_cfg, args, model_name="SHLNN", num_inputs=pca_dim)
    shlnn_trainer.train(
        num_epochs=PAPER_MAX_EPOCHS,
        train_loader=pca_train_loader,
        val_loader=pca_val_loader,
        test_loader=pca_test_loader,   # evaluate trên fold bị giữ lại
        results_dir=shlnn_result_dir,
        patience=PATIENCE,
    )

    # --------------------------------------------------------
    # 6. Đọc test accuracy từ final_metrics.json
    # --------------------------------------------------------
    metrics_path = os.path.join(shlnn_result_dir, "final_metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        final_metrics = json.load(f)

    test_acc  = final_metrics.get("test_acc", 0.0)
    test_loss = final_metrics.get("test_loss", 0.0)

    print(f"\nFOLD {fold_idx} RESULT  →  test_acc={test_acc*100:.2f}%  |  test_loss={test_loss:.5f}")

    # --------------------------------------------------------
    # 7. Ghi fold summary
    # --------------------------------------------------------
    fold_summary = {
        "fold"              : fold_idx,
        "test_fold"         : f"Fold_{fold_idx}",
        "pca_dimensions"    : pca_dim,
        "arcnn_best_epoch"  : int(arcnn_trainer.best_epoch + 1) if arcnn_trainer.best_epoch is not None else None,
        "shlnn_best_epoch"  : int(shlnn_trainer.best_epoch + 1) if shlnn_trainer.best_epoch is not None else None,
        "test_acc"          : float(test_acc),
        "test_loss"         : float(test_loss),
    }

    with open(os.path.join(fold_result_dir, "fold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    # --------------------------------------------------------
    # 8. Dọn PCA intermediate để tiết kiệm disk
    # --------------------------------------------------------
    del arcnn_trainer, shlnn_trainer
    del pca_train_loader, pca_val_loader, pca_test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if os.path.isdir(pca_dir):
        shutil.rmtree(pca_dir, ignore_errors=True)

    return fold_summary


def main():
    cfgs, args = get_agrs_parser()
    cfgs, args = prepare_runtime(cfgs, args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("ArcNN  —  K-Fold Cross Validation Pipeline")
    print("=" * 70)
    print(f"  K-folds  : {N_FOLDS}")
    print(f"  Batch    : {PAPER_BATCH_SIZE}")
    print(f"  Epochs   : {PAPER_MAX_EPOCHS}")
    print(f"  PCA EVR  : {PCA_EVR}")
    print(f"  ArcNN LR : {AR_CNN_LR}")
    print(f"  SHLNN LR : {SHLNN_LR}")
    print(f"  Patience : {PATIENCE}")
    print(f"  Scratch  : {SCRATCH_ROOT}")
    print(f"  Output   : {FINAL_ROOT}")
    print("=" * 70)
    print("\nCấu trúc K-fold:")
    for i in range(1, N_FOLDS + 1):
        others = [str(j) for j in range(1, N_FOLDS + 1) if j != i]
        print(f"  Vòng {i}: test=Fold_{i}  |  train+val=Fold_{'+'.join(others)}")
    print()

    # get_dataloader_folds tự động tạo đúng 5 bộ loader theo K-fold chuẩn
    loaders = get_dataloader(cfgs, args)

    if len(loaders) != N_FOLDS:
        print(f"WARNING: tìm thấy {len(loaders)} fold, không phải {N_FOLDS}.")

    all_summaries = []

    for fold_idx, fold_loaders in enumerate(loaders, start=1):
        summary = run_one_fold(fold_idx, fold_loaders, cfgs, args)
        all_summaries.append(summary)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Tổng hợp kết quả 5 vòng
    # --------------------------------------------------------
    accs = [s["test_acc"] for s in all_summaries]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    final_summary = {
        "all_folds"    : all_summaries,
        "mean_test_acc": mean_acc,
        "std_test_acc" : std_acc,
        "paper_target" : 0.9988,
    }

    with open(os.path.join(FINAL_ROOT, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("5-FOLD CROSS VALIDATION COMPLETED")
    print("=" * 70)
    for s in all_summaries:
        print(f"  Fold {s['fold']} (test={s['test_fold']}): "
              f"acc={s['test_acc']*100:.2f}%  |  loss={s['test_loss']:.5f}")
    print(f"\n  Mean accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Paper target  : 99.88%")
    print(f"\n  Results saved : {FINAL_ROOT}")


if __name__ == "__main__":
    main()