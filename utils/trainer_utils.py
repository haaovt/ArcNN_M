"""
trainer_utils.py
----------------
Training loop cho ArcNN và SHLNN.

Các điểm đã sửa theo paper (đã đối chiếu với file paper thật — Yan,
Li, Duan, "A Simplified Current Feature Extraction and Deployment
Method for DC Series Arc Fault Detection", IEEE TIE 2024):
- RMSprop.
- ArcNN: CrossEntropyLoss (paper Section IV-A, target là class-index).
- SHLNN: mặc định BCELoss + Sigmoid output (paper Section III-C, target
  one-hot [0,1]/[1,0]) — KHÁC với ArcNN. Config hiện tại của bạn
  (config_stage2.yaml) ghi `loss_type: CrossEntropy` cho SHLNN, xung
  đột với paper — có thể tắt Sigmoid/BCE và quay về CrossEntropy qua
  cfgs['SHLNN']['sigmoid_output']=False. Xem model_utils.py.SHLNN,
  BaseTrainer._uses_sigmoid_bce() và _prepare_target() ở dưới.
- Early stopping theo validation loss, patience=3 — ĐÃ XÁC NHẬN khớp
  paper ("When the loss stops decreasing for more than three epochs,
  the training is stopped").
- Paper: batch_size=64, epochs=50 (2 giá trị này nằm ở train.py/config,
  không có trong các file utils; xem sweep_utils.py đã cập nhật
  batch_size=64 cố định theo paper).
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

        # Lưu lại để _prepare_target / _compute_class_weights / train()
        # biết cách format target và chọn loss đúng cho từng model.
        self.model_name = model_name
        # (self.model được gán ở khối if/elif ngay dưới; _uses_sigmoid_bce
        # được gọi SAU khi self.model tồn tại, xem property bên dưới.)

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

        # SỬA: ArcNN dùng CrossEntropyLoss (paper Section IV-A, target là
        # class-index). SHLNN dùng BCELoss vì model_utils.SHLNN giờ có
        # Sigmoid ở output theo mặc định (paper Section III-C) —
        # CrossEntropyLoss không hợp với input đã qua sigmoid (nó tự áp
        # log_softmax lên input, giả định input là logits chưa qua
        # activation nào). Việc chọn BCE hay CrossEntropy được quyết
        # định qua self._uses_sigmoid_bce (đọc từ chính model đã tạo ở
        # trên, tôn trọng cfgs['SHLNN']['sigmoid_output'] nếu bạn tắt
        # nó đi — xem model_utils.py.SHLNN).
        # Xem _prepare_target() để biết cách format target tương ứng.
        if self._uses_sigmoid_bce():
            self.loss_type = nn.BCELoss()
        else:
            self.loss_type = nn.CrossEntropyLoss()

        if self.cuda:
            self.model.cuda()

        self.best_epoch = None
        self.best_val_loss = float("inf")
        self.best_state = None

    def _uses_sigmoid_bce(self):
        """
        True nếu model hiện tại có output đã qua Sigmoid (mặc định cho
        SHLNN theo paper Section III-C) -> cần BCELoss + target one-hot.
        False nếu output là logits -> CrossEntropyLoss + target class-index.

        Đọc trực tiếp từ `self.model.use_sigmoid_output` (đặt ở
        model_utils.py.SHLNN.__init__ theo cfgs['SHLNN']['sigmoid_output'],
        mặc định True) thay vì chỉ dựa vào model_name, để việc tắt
        Sigmoid qua config (xem "XUNG ĐỘT VỚI CONFIG" trong
        model_utils.py) có tác dụng thật sự ở đây, không bị bỏ qua.
        """
        return self.model_name == "SHLNN" and getattr(
            self.model, "use_sigmoid_output", True
        )

    def _prepare_target(self, all_y):
        """
        Chuẩn hoá target cho đúng loss/activation của từng model.

        - ArcNN (hoặc SHLNN với sigmoid_output=False): output là logits,
          dùng CrossEntropyLoss -> target giữ nguyên dạng class-index
          (long), như trước.
        - SHLNN với sigmoid_output=True (mặc định): output đã qua
          Sigmoid (paper Section III-C), dùng BCELoss -> target phải
          cùng shape với output và ở dạng one-hot float, đúng như paper
          mô tả nhãn: "[0,1] and [1,0] for dc arc and normal,
          respectively".

        all_y (từ DataLoader) luôn là class-index (0/1) bất kể model,
        vì CustomDataset/PCADataset không đổi theo model — việc chuyển
        sang one-hot chỉ cần làm ở đây, ngay trước khi tính loss.
        """
        if self._uses_sigmoid_bce():
            return torch.nn.functional.one_hot(all_y, num_classes=2).float()
        return all_y

    def _compute_class_weights(self, train_loader):
        """
        Tính class weights từ train_loader để xử lý class imbalance.

        SỬA (đã đối chiếu với file paper thật):
        Bài paper KHÔNG hề mô tả bất kỳ kỹ thuật xử lý mất cân bằng lớp
        nào (không class weight, không oversampling, không focal loss).
        Hơn nữa, Table II của paper cho thấy dataset của paper khá CÂN
        BẰNG: tổng training samples ~15957, trong đó arc ≈ 8629 (54%)
        và normal ≈ 7328 (46%) — không hề lệch như ví dụ "Normal=98%,
        Arc=2%" mà comment cũ của hàm này minh hoạ (con số đó chỉ là ví
        dụ giả định, không lấy từ paper). File CSV mẫu bạn gửi cũng cho
        kết quả tương tự (1090 arc / 863 normal ≈ 56/44 cho 1 file).

        Vì vậy, mặc định của hàm này đổi thành 'none' (KHÔNG reweight)
        để khớp đúng những gì paper thực sự làm. Vẫn giữ 2 lựa chọn
        khác qua cfgs['class_weight_scheme'] để bạn tự dùng nếu dataset
        thật của bạn có fold/file lệch nhãn nhiều hơn paper:
            'none'      (mặc định — khớp đúng paper, không reweighting)
            'balanced'  (n/(2*count) kiểu sklearn, không hyperparameter tự chế)
            'sqrt_soft' (công thức tự chế cũ: sqrt + clamp 10x + normalize)

        LƯU Ý khi dùng 'balanced'/'sqrt_soft' với SHLNN: BCELoss áp
        weight theo shape của OUTPUT (mỗi sample đều bị nhân weight[0]
        cho unit "normal" và weight[1] cho unit "arc"), khác với
        CrossEntropyLoss (áp weight[y_i] theo đúng NHÃN THẬT của từng
        sample). Hai cách này không tương đương về ý nghĩa thống kê —
        nếu cần reweight SHLNN đúng kiểu per-sample, nên cân nhắc
        oversampling dữ liệu thay vì dùng `weight=` của BCELoss.

        SỬA LỖI CHIA CHO 0:
        Nếu một lớp vắng mặt hoàn toàn trong train_loader (dễ xảy ra
        khi K-fold/train-val split chưa cân bằng nhãn giữa các fold),
        total/(2*count) sẽ chia cho 0 -> inf. Bản cũ vô tình "che" lỗi
        này bằng torch.clamp(max=10), khiến người dùng không biết fold
        đang gặp vấn đề dữ liệu. Giờ phát hiện và cảnh báo rõ, đồng
        thời fallback về weight=1 (không reweight) cho lần gọi đó thay
        vì âm thầm dùng một con số bị chặn trần không phản ánh đúng
        thực tế.
        """
        label_counts = torch.zeros(2)

        for _, all_y in train_loader:
            for c in range(2):
                label_counts[c] += (all_y == c).sum()

        total = label_counts.sum()
        scheme = self.cfgs.get("class_weight_scheme", "none")

        if scheme == "none":
            weights = torch.ones(2)
            print(f"Class weights: tắt (scheme='none') — dùng weight=1 cho mọi lớp.")
            return weights

        if (label_counts == 0).any():
            missing = [c for c in range(2) if label_counts[c] == 0]
            print(
                f"[CẢNH BÁO] Lớp {missing} không xuất hiện trong tập train "
                f"của lần gọi train() này -> bỏ qua class weighting "
                f"(weight=1) để tránh chia cho 0. Đây thường là dấu hiệu "
                f"fold/train-val đang mất cân bằng nhãn nặng — nên kiểm "
                f"tra lại bước chia K-fold trong preprocess_utils.py."
            )
            return torch.ones(2)

        if scheme == "sqrt_soft":
            # Công thức cũ — giữ lại để so sánh, KHÔNG còn là mặc định.
            weights = torch.sqrt(total / (2 * label_counts))
            weights = torch.clamp(weights, max=10.0)
            weights = weights / weights.mean()
        else:  # "balanced" (mặc định) — công thức chuẩn kiểu sklearn
            weights = total / (2 * label_counts)

        print(
            f"Class weights (scheme={scheme}) — Normal: {weights[0]:.3f} | "
            f"Arc: {weights[1]:.3f}  "
            f"(Normal: {label_counts[0].long()} samples | "
            f"Arc: {label_counts[1].long()} samples)"
        )

        return weights

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
            # SỬA: target phải khớp loss của từng model (xem _prepare_target).
            loss = self.loss_type(pred, self._prepare_target(all_y))

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
            # SỬA: target phải khớp loss của từng model (xem _prepare_target).
            loss = self.loss_type(pred, self._prepare_target(all_y))

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

        ĐÃ XÁC NHẬN VỚI PAPER: `patience=3` (giá trị default của tham
        số này, và cũng là default của class EarlyStopping) khớp đúng
        paper Section IV-A: "When the loss stops decreasing for more
        than three epochs, the training is stopped". Nếu train.py hoặc
        config nào đó truyền một giá trị patience khác 3, giờ có thể
        khẳng định điều đó đang lệch khỏi paper (trước đây mình chưa
        xác nhận được nên chỉ nêu là cần kiểm tra). Paper cũng nêu rõ
        num_epochs=50 — tham số này nằm ở train.py/config, không có
        trong utils/, bạn nên xác nhận riêng.
        """
        if results_dir is None:
            raise ValueError("results_dir không được None.")

        os.makedirs(
            os.path.join(results_dir, "ckpts"),
            exist_ok=True,
        )

        history = []

        # Tính class weights từ train_loader để xử lý class imbalance
        print("Tính class weights...")
        class_weights = self._compute_class_weights(train_loader)
        if self.cuda:
            class_weights = class_weights.cuda()
        # SỬA: loss phải khớp model (xem __init__ / model_utils.SHLNN).
        # Mặc định class_weight_scheme='none' -> class_weights toàn 1.0,
        # tương đương không reweight (đúng với paper — xem docstring
        # _compute_class_weights).
        if self._uses_sigmoid_bce():
            self.loss_type = nn.BCELoss(weight=class_weights)
        else:
            self.loss_type = nn.CrossEntropyLoss(weight=class_weights)

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

            # Dừng nếu validation loss không giảm quá `patience` epoch
            # liên tiếp (xem lưu ý về giá trị patience ở docstring trên).
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
