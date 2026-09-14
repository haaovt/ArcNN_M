import torch
import numpy as np
from torch.utils.data import Dataset
import os
import bisect


class CustomDataset(Dataset):
    """
    Đọc các cặp file <name>_x.npy / <name>_y.npy bằng mmap.

    SỬA (hỗ trợ chia train/val theo FILE, không theo window):
    Trước đây CustomDataset chỉ nhận `dataset_dir` và tự quét TOÀN BỘ
    file trong thư mục đó. init_utils.py vì vậy buộc phải tạo 1
    CustomDataset cho cả một Fold_X rồi mới Subset theo window-index để
    chia train/val -> làm mất ranh giới file (xem lỗi rò rỉ dữ liệu mô
    tả trong init_utils.py).

    Thêm tham số `file_pairs` để init_utils.py có thể tự chọn trước một
    tập con file (x_path, y_path) rồi build CustomDataset thẳng từ đó.
    Nhờ vậy một file luôn thuộc trọn vẹn một dataset (train HOẶC val),
    không bao giờ bị cắt đôi ở mức window.

    `dataset_dir` vẫn được giữ nguyên hành vi cũ (quét cả thư mục) để
    tương thích ngược — dùng cho các trường hợp không cần chia
    train/val (ví dụ dataset của fold dùng làm test).
    """

    def __init__(self, dataset_dir=None, file_pairs=None, file_loader=None, transform=None, cfgs=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        if file_pairs is not None:
            # Danh sách (x_path, y_path) đã chọn sẵn ở mức FILE.
            file_pairs = sorted(file_pairs, key=lambda pair: pair[0])
            self.x_paths = [p[0] for p in file_pairs]
            self.y_paths = [p[1] for p in file_pairs]
        elif dataset_dir is not None:
            self.x_paths, self.y_paths = self._scan_dir(dataset_dir)
        else:
            raise ValueError(
                "CustomDataset cần một trong hai: dataset_dir hoặc file_pairs."
            )

        self.cumulative_sizes = []
        total_chunks = 0

        for p in self.x_paths:
            # Chỉ đọc header để lấy kích thước (0MB RAM)
            arr_view = np.load(p, mmap_mode='r')
            total_chunks += arr_view.shape[0]
            self.cumulative_sizes.append(total_chunks)

        self.total_samples = total_chunks

    @staticmethod
    def _scan_dir(dataset_dir):
        """Quét thư mục, trả về (x_paths, y_paths) đã sort — không load dữ liệu."""
        x_paths = sorted([
            os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir)
            if p.lower().endswith('_x.npy')
        ])
        y_paths = [p.replace('_x.npy', '_y.npy') for p in x_paths]
        return x_paths, y_paths

    @staticmethod
    def list_file_pairs(dataset_dir):
        """
        Trả về [(x_path, y_path), ...] trong dataset_dir mà KHÔNG load
        dữ liệu (chỉ liệt kê tên file).

        Dùng ở init_utils.py để chia train/val theo FILE trước khi tạo
        CustomDataset, thay vì chia theo window-index sau khi đã gộp.
        """
        x_paths, y_paths = CustomDataset._scan_dir(dataset_dir)
        return list(zip(x_paths, y_paths))

    def __getitem__(self, index):
        file_idx = bisect.bisect_right(self.cumulative_sizes, index)

        if file_idx == 0:
            rel_idx = index
        else:
            rel_idx = index - self.cumulative_sizes[file_idx - 1]

        x_mmap = np.load(self.x_paths[file_idx], mmap_mode='r')
        y_mmap = np.load(self.y_paths[file_idx], mmap_mode='r')

        x = np.array(x_mmap[rel_idx])
        y = np.array(y_mmap[rel_idx])

        if self.transform is not None:
            x = self.transform(x)

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.total_samples


class PCADataset(Dataset):
    def __init__(self, feature_path, labels_path):
        super().__init__()
        self.features = np.load(feature_path)
        self.labels = np.load(labels_path)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
