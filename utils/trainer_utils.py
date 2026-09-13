"""
trainer_utils.py
----------------
Training loop cho ArcNN và SHLNN.

Các điểm đã sửa theo paper:
- RMSprop.
- CrossEntropyLoss.
- Early stopping theo validation loss.
- KHÔNG test mỗi epoch.
- Lưu duy nhất best checkpoint.
- Khôi phục best model trước khi test/extract feature.
- Giảm mạnh số file checkpoint -> tránh đầy /kaggle/working.
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils.model_utils import ArcNN, SHLNN


class EarlyStopping:
    """Early stopping dựa trên validation loss."""

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def step(self, val_loss):
        """
        Trả về True nếu cần dừng.

        Nếu validation loss giảm -> reset counter.
        Nếu không giảm đủ -> tăng counter.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class BaseTrainer:
    """
    Trainer dùng chung cho ArcNN và SHLNN.

    model_name:
        ArcNN hoặc SHLNN

    num_inputs:
        Chỉ cần thiết cho SHLNN sau PCA.
    """

    def __init__(self, cfgs, args, model_name=None, num_inputs=None):
        self.cfgs = cfgs
        self.cuda = bool(
            getattr(args, "cuda", False)
            and torch.cuda.is_available()
        )

        if model_name is None:
            model_name = cfgs.get("model", "ArcNN")

        if model_name == "ArcNN":
            self.model = ArcNN(cfgs)
        elif model_name == "SHLNN":
            self.model = SHLNN(
                cfgs,
                num_inputs=num_inputs,
            )
        else:
            raise ValueError(
                f"Model không hỗ trợ: {model_name}"
            )

        # Paper dùng RMSprop.
        learning_rate = float(
            cfgs.get(
                "learning_rate",
                2e-4 if model_name == "ArcNN" else 1e-3,
            )
        )

        weight_decay = float(
            cfgs.get("weight_decay", 0.0)
        )

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Paper dùng cross-entropy.
        self.loss_type = nn.CrossEntropyLoss()

        if self.cuda:
            self.model.cuda()

        self.best_epoch = None
        self.best_val_loss = float("inf")
        self.best_state = None

    def predict(self, x):
        return self.model(x)

    def train_step(self, train_loader):
        """Một epoch training."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for all_x, all_y in train_loader:
            if self.cuda:
                all_x = all_x.cuda(non_blocking=True)
                all_y = all_y.cuda(non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.predict(all_x)
            loss = self.loss_type(pred, all_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * all_x.size(0)

            pred_classes = pred.argmax(dim=1)
            total_correct += (
                pred_classes.eq(all_y).sum().item()
            )
            total_samples += all_x.size(0)

        if total_samples == 0:
            return {
                "loss": 0.0,
                "acc": 0.0,
            }

        return {
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples,
        }

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Validation/test.

        Test chỉ nên gọi SAU KHI training kết thúc và best model
        đã được restore.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for all_x, all_y in loader:
            if self.cuda:
                all_x = all_x.cuda(non_blocking=True)
                all_y = all_y.cuda(non_blocking=True)

            pred = self.predict(all_x)
            loss = self.loss_type(pred, all_y)

            total_loss += loss.item() * all_x.size(0)
            total_correct += (
                pred.argmax(dim=1).eq(all_y).sum().item()
            )
            total_samples += all_x.size(0)

        if total_samples == 0:
            return 0.0, 0.0

        return (
            total_correct / total_samples,
            total_loss / total_samples,
        )

    def save_best_checkpoint(self, results_dir, epoch):
        """
        Chỉ ghi đè một file best_ckpt.pth.

        Đây là thay đổi quan trọng để không tạo:
            Epoch_1_ckpt.pth
            Epoch_2_ckpt.pth
            Epoch_3_ckpt.pth
            ...

        mà chỉ giữ:
            best_ckpt.pth
        """
        ckpt_dir = os.path.join(results_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            ckpt_dir,
            "best_ckpt.pth",
        )

        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_loss": self.best_val_loss,
        }

        torch.save(state, checkpoint_path)
        return checkpoint_path

    def load_best_checkpoint(self, results_dir):
        """Load best checkpoint trước khi test/extract feature."""
        checkpoint_path = os.path.join(
            results_dir,
            "ckpts",
            "best_ckpt.pth",
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Không tìm thấy best checkpoint: {checkpoint_path}"
            )

        state = torch.load(
            checkpoint_path,
            map_location="cuda" if self.cuda else "cpu",
            weights_only=False,
        )

        self.model.load_state_dict(state["model"])

        if "optimizer" in state:
            self.optimizer.load_state_dict(
                state["optimizer"]
            )

        self.best_epoch = state.get("epoch")
        self.best_val_loss = state.get(
            "val_loss",
            self.best_val_loss,
        )

        return self.best_epoch

    def train(
        self,
        num_epochs,
        train_loader,
        val_loader,
        test_loader=None,
        results_dir=None,
        cur_epoch=0,
        patience=3,
    ):
        """
        Train và chọn model theo validation loss.

        QUAN TRỌNG:
        Không evaluate test trong vòng epoch.
        Điều này tránh dùng test set để quan sát training.
        """
        if results_dir is None:
            raise ValueError("results_dir không được None.")

        os.makedirs(
            os.path.join(results_dir, "ckpts"),
            exist_ok=True,
        )

        history = []

        early_stopping = EarlyStopping(
            patience=patience
        )

        iterator = tqdm(
            range(cur_epoch, num_epochs),
            total=max(num_epochs - cur_epoch, 0),
            unit="epoch",
            desc="Training",
        )

        for epoch in iterator:
            train_metrics = self.train_step(
                train_loader
            )

            val_acc, val_loss = self.evaluate(
                val_loader
            )

            stats = {
                "epoch": int(epoch + 1),
                "train_loss": float(
                    train_metrics["loss"]
                ),
                "train_acc": float(
                    train_metrics["acc"]
                ),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }

            history.append(stats)

            tqdm.write(
                f"Epoch {epoch + 1:02d}/{num_epochs} | "
                f"train_loss={stats['train_loss']:.5f} | "
                f"train_acc={stats['train_acc']:.4f} | "
                f"val_loss={stats['val_loss']:.5f} | "
                f"val_acc={stats['val_acc']:.4f}"
            )

            # Lưu best model duy nhất.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

                # Giữ state trên CPU để không chiếm GPU.
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                self.save_best_checkpoint(
                    results_dir,
                    epoch,
                )

            # Paper: dừng nếu validation loss không giảm >3 epoch.
            if early_stopping.step(val_loss):
                tqdm.write(
                    "Early stopping: validation loss "
                    "không cải thiện."
                )
                break

        # Khôi phục best state ngay trong memory.
        if self.best_state is not None:
            self.model.load_state_dict(
                self.best_state
            )

        # Ghi history nhỏ gọn.
        history_path = os.path.join(
            results_dir,
            "training_history.json",
        )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
            )

        # Test CHỈ sau khi đã chọn best validation model.
        if test_loader is not None:
            test_acc, test_loss = self.evaluate(
                test_loader
            )

            final_metrics = {
                "best_epoch": (
                    None
                    if self.best_epoch is None
                    else int(self.best_epoch + 1)
                ),
                "best_val_loss": float(
                    self.best_val_loss
                ),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            }

            with open(
                os.path.join(
                    results_dir,
                    "final_metrics.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    final_metrics,
                    f,
                    indent=2,
                )

            tqdm.write(
                f"BEST epoch={final_metrics['best_epoch']} | "
                f"test_loss={test_loss:.5f} | "
                f"test_acc={test_acc:.4f}"
            )

        return history

    @torch.no_grad()
    def extract_features(
        self,
        loader,
        save_path,
    ):
        """
        Extract ArcNN features 1024 chiều.

        Dùng memmap để không phải giữ toàn bộ feature matrix
        trong RAM cùng lúc.
        """
        self.model.eval()

        total_samples = len(loader.dataset)
        feature_dim = 1024

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True,
        )

        # Memmap: dữ liệu được ghi dần ra disk.
        # Dùng file tạm .dat để np.save(save_path, ...) tạo đúng
        # tên file .npy, tránh lỗi thành train_features.npy.npy.
        feature_tmp_path = save_path + ".tmp.dat"

        features = np.memmap(
            feature_tmp_path,
            dtype="float32",
            mode="w+",
            shape=(total_samples, feature_dim),
        )

        labels_path = save_path.replace(
            "features.npy",
            "labels.npy",
        )
        label_tmp_path = labels_path + ".tmp.dat"

        labels = np.memmap(
            label_tmp_path,
            dtype="int64",
            mode="w+",
            shape=(total_samples,),
        )

        current_idx = 0

        for all_x, all_y in tqdm(
            loader,
            desc=f"Extract -> {os.path.basename(save_path)}",
        ):
            if self.cuda:
                all_x = all_x.cuda(
                    non_blocking=True
                )

            batch_features = (
                self.model(
                    all_x,
                    extract_features=True,
                )
                .cpu()
                .numpy()
            )

            batch_size = batch_features.shape[0]

            features[
                current_idx:current_idx + batch_size
            ] = batch_features

            labels[
                current_idx:current_idx + batch_size
            ] = all_y.numpy()

            current_idx += batch_size

        features.flush()
        labels.flush()

        # Đổi tên memmap -> .npy thật bằng cách ghi lại array.
        # Ở bước này feature đã nằm trên disk; đọc mmap nên không
        # cần giữ toàn bộ matrix trong RAM.
        np.save(
            save_path,
            np.asarray(features),
        )
        np.save(
            labels_path,
            np.asarray(labels),
        )

        # Xóa file memmap tạm.
        try:
            os.remove(feature_tmp_path)
        except FileNotFoundError:
            pass

        try:
            os.remove(label_tmp_path)
        except FileNotFoundError:
            pass

        print(
            f"Feature extraction done: "
            f"{total_samples} x {feature_dim}"
        )

    def load_ckpt(self, checkpoint_path):
        """
        Tương thích với checkpoint cũ của repo.
        """
        state = torch.load(
            checkpoint_path,
            map_location="cuda" if self.cuda else "cpu",
            weights_only=False,
        )

        self.model.load_state_dict(
            state["model"]
        )

        if "optimizer" in state:
            self.optimizer.load_state_dict(
                state["optimizer"]
            )

        return state.get("epoch", 0)