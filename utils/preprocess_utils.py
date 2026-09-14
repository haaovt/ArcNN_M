"""
preprocess_utils.py
-------------------
Tiền xử lý dữ liệu cho ArcNN — chia TRAIN/VAL/TEST một lần
(KHÔNG còn K-fold, theo đúng cách bài báo mô tả ở Bảng II).

Mục tiêu:
1. Đọc Current_A và Voltage_V từ CSV.
2. Chia dữ liệu thành 3 tập TRAIN/VAL/TEST theo tỉ lệ ~64/16/20%
   — đúng tỉ lệ thực tế trong Bảng II của bài báo
   (15957/3985/4981 trên tổng 24918 mẫu ≈ 64.0% / 16.0% / 20.0%).
3. Ưu tiên chia theo FILE (mỗi file CSV = một lần thu thí nghiệm)
   để tránh rò rỉ dữ liệu (leakage) giữa các tập — các window của
   cùng một file KHÔNG bị chia sang nhiều tập khác nhau.
4. Nếu số file CSV quá ít (<3, không đủ để chia 3 tập theo file),
   tự động chuyển sang chia THEO THỜI GIAN bên trong từng file
   (64% đầu -> train, 16% giữa -> val, 20% cuối -> test) để vẫn
   tránh trộn lẫn các window liền kề giữa các tập.
5. Tạo window dài 512 điểm, không overlap.
6. Gán nhãn theo tiêu chí điện áp ổn định của paper: hồ quang khi
   điện áp nằm trong khoảng ĐÓNG 15-20V (không chỉ >=15V).
7. Lọc dòng điện trong dải 90-110 kHz (xấp xỉ CBF phần cứng).
8. Min-Max về [0, 1].
9. Lưu X/Y theo từng file để CustomDataset có thể dùng mmap_mode='r'.

CẤU TRÚC THƯ MỤC ĐẦU RA:
    output_dir/
        train/<file>_x.npy, <file>_y.npy
        val/<file>_x.npy,   <file>_y.npy
        test/<file>_x.npy,  <file>_y.npy

LƯU Ý VỀ PAPER:
Paper mô tả CBF (Composite Bandpass Filter) phần cứng với dải 90-110 kHz.
Code hiện tại không có mô hình CBF phần cứng, nên Butterworth
digital band-pass dưới đây chỉ là phép xấp xỉ phần lọc 90-110 kHz.
Không nên gọi đây là CBF phần cứng chính xác.
"""

import os
import random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm


# =========================
# Các tham số theo paper
# =========================
FS = 250_000          # Sampling frequency: 250 kHz
LOWCUT = 90_000       # Lower cutoff: 90 kHz
HIGHCUT = 110_000     # Upper cutoff: 110 kHz
FILTER_ORDER = 4
SEQ_LEN = 512         # Paper sử dụng 512 điểm / sample
ARC_VOLTAGE_MIN = 15
ARC_VOLTAGE_MAX = 20

# Tỉ lệ Train/Val/Test — khớp đúng Bảng II của paper:
# 15957/3985/4981 trên tổng 24918 mẫu ≈ 64.0% / 16.0% / 20.0%.
TRAIN_RATIO = 0.64
VAL_RATIO = 0.16
TEST_RATIO = 0.20  # = 1 - TRAIN_RATIO - VAL_RATIO


def butter_bandpass_filter(
    data,
    lowcut=LOWCUT,
    highcut=HIGHCUT,
    fs=FS,
    order=FILTER_ORDER,
):
    """
    Xấp xỉ phần band-pass 90-110 kHz bằng Butterworth digital filter.

    Paper dùng CBF phần cứng; hàm này chỉ tái tạo đặc tính dải tần
    cần thiết khi dữ liệu đầu vào là raw CSV.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def normalize_minmax(signal):
    """Đưa một sample về khoảng [0, 1] — đúng Eq.(4) của paper."""
    signal = np.asarray(signal, dtype=np.float32)

    s_min = np.min(signal)
    s_max = np.max(signal)

    if s_max > s_min:
        return (signal - s_min) / (s_max - s_min)

    # Trường hợp tín hiệu phẳng.
    return np.zeros_like(signal, dtype=np.float32)


def make_samples_from_csv(file_path, seq_len=SEQ_LEN):
    """
    Đọc một CSV và tạo toàn bộ sample 512 điểm (theo thứ tự thời gian).

    Label theo tiêu chí điện áp trong paper (mục II.B):
    hồ quang ổn định có điện áp trong khoảng ĐÓNG 15-20V. Nếu có ít
    nhất một điểm trong window rơi vào khoảng này -> label arc = 1.

    Voltage không được đưa vào ArcNN, chỉ dùng để gán nhãn.
    """
    df = pd.read_csv(file_path)

    if "Current_A" not in df.columns or "Voltage_V" not in df.columns:
        raise ValueError(
            f"{file_path} phải có hai cột Current_A và Voltage_V"
        )

    current = df["Current_A"].to_numpy(dtype=np.float32)
    voltage = df["Voltage_V"].to_numpy(dtype=np.float32)

    samples = []
    labels = []

    # Không overlap: mỗi window có đúng 512 điểm, giữ nguyên thứ tự
    # thời gian (quan trọng để có thể chia chronological khi cần).
    for start in range(0, len(current) - seq_len + 1, seq_len):
        chunk_current = current[start:start + seq_len]
        chunk_voltage = voltage[start:start + seq_len]

        # Paper (Section II.B): hồ quang ổn định có điện áp trong
        # khoảng ĐÓNG 15-20V. Chỉ cần MỘT điểm trong window rơi vào
        # khoảng này -> có hồ quang xảy ra trong window đó.
        abs_voltage = np.abs(chunk_voltage)
        in_arc_range = (abs_voltage >= ARC_VOLTAGE_MIN) & (abs_voltage <= ARC_VOLTAGE_MAX)
        label = int(np.any(in_arc_range))

        # Band-pass 90-110 kHz, sau đó Min-Max.
        filtered_current = butter_bandpass_filter(chunk_current)
        normalized_current = normalize_minmax(filtered_current)

        # Shape: (1, 512)
        samples.append(
            normalized_current.reshape(1, seq_len).astype(np.float32)
        )
        labels.append(label)

    if not samples:
        return None, None

    return (
        np.stack(samples, axis=0),
        np.asarray(labels, dtype=np.int64),
    )


def _save_split(output_dir, split_name, base_name, x, y):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, f"{base_name}_x.npy"), x)
    np.save(os.path.join(split_dir, f"{base_name}_y.npy"), y)


def process_file_whole(args):
    """
    Worker: xử lý một CSV và lưu TOÀN BỘ file vào một tập duy nhất
    (train/val/test) — dùng khi có đủ file để chia theo FILE.
    """
    split_name, file_path, output_dir, seq_len = args

    x, y = make_samples_from_csv(file_path, seq_len=seq_len)
    if x is None:
        return split_name, os.path.basename(file_path), 0

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    _save_split(output_dir, split_name, base_name, x, y)

    return split_name, base_name, len(y)


def process_file_chronological(args):
    """
    Worker: xử lý một CSV rồi chia CHÍNH file đó theo thời gian
    thành train/val/test (64%/16%/20% đầu->cuối) — dùng khi số file
    quá ít để chia theo FILE mà vẫn cần tránh trộn window liền kề
    giữa các tập.
    """
    file_path, output_dir, seq_len, train_ratio, val_ratio = args

    x, y = make_samples_from_csv(file_path, seq_len=seq_len)
    if x is None:
        return os.path.basename(file_path), {"train": 0, "val": 0, "test": 0}

    n = len(y)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    counts = {}

    for split_name, (s, e) in (
        ("train", (0, n_train)),
        ("val", (n_train, n_train + n_val)),
        ("test", (n_train + n_val, n)),
    ):
        if e > s:
            _save_split(output_dir, split_name, base_name, x[s:e], y[s:e])
        counts[split_name] = max(0, e - s)

    return base_name, counts


def process_raw_csv_to_train_val_test(
    input_dir,
    output_dir,
    seq_len=SEQ_LEN,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    seed=42,
    num_workers=None,
):
    """
    Chia CSV thành TRAIN/VAL/TEST (không còn K-fold).

    - Nếu có >= 3 file CSV: chia theo FILE (mỗi file thuộc trọn vẹn
      một tập) — cách này khớp với cách paper thu thập nhiều lần thí
      nghiệm riêng biệt (Bảng II / Hình 6) và tránh leakage tuyệt đối.
    - Nếu có < 3 file CSV: không đủ để chia theo file, tự động
      chuyển sang chia THEO THỜI GIAN bên trong từng file.

    output_dir nên nằm ở scratch disk của Kaggle:
        /kaggle/tmp/arcnn/dataset/MyData
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy CSV trong: {input_dir}"
        )

    random.seed(seed)
    random.shuffle(csv_files)

    n = len(csv_files)

    if n >= 3:
        # ---- Chia theo FILE ----
        n_train = max(1, round(train_ratio * n))
        n_train = min(n_train, n - 2)  # chừa ít nhất 1 file cho val và 1 cho test
        n_val = max(1, round(val_ratio * n))
        n_val = min(n_val, n - n_train - 1)

        train_files = csv_files[:n_train]
        val_files = csv_files[n_train:n_train + n_val]
        test_files = csv_files[n_train + n_val:]

        print(
            f"Chia {n} file CSV theo FILE: "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)} "
            f"(seed={seed})"
        )

        tasks = (
            [("train", f, output_dir, seq_len) for f in train_files]
            + [("val", f, output_dir, seq_len) for f in val_files]
            + [("test", f, output_dir, seq_len) for f in test_files]
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_file_whole, tasks),
                    total=len(tasks),
                    desc="Preprocessing (split theo file)",
                )
            )

        totals = {"train": 0, "val": 0, "test": 0}
        for split_name, _, count in results:
            totals[split_name] += count

    else:
        # ---- Quá ít file: chia THEO THỜI GIAN bên trong từng file ----
        print(
            f"CẢNH BÁO: chỉ có {n} file CSV (<3), không đủ để chia "
            f"train/val/test theo FILE. Chuyển sang chia THEO THỜI GIAN "
            f"bên trong từng file: {train_ratio*100:.0f}% đầu -> train, "
            f"{val_ratio*100:.0f}% giữa -> val, phần còn lại -> test."
        )

        tasks = [
            (f, output_dir, seq_len, train_ratio, val_ratio)
            for f in csv_files
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_file_chronological, tasks),
                    total=len(tasks),
                    desc="Preprocessing (split theo thời gian)",
                )
            )

        totals = {"train": 0, "val": 0, "test": 0}
        for _, counts in results:
            for k in totals:
                totals[k] += counts[k]

    print("\n===== PREPROCESSING DONE =====")
    print(f"CSV files : {n}")
    print(f"Train     : {totals['train']} samples")
    print(f"Val       : {totals['val']} samples")
    print(f"Test      : {totals['test']} samples")
    print(f"Output    : {output_dir}")

    return totals


def auto_find_kaggle_input():
    """
    Tìm thư mục input chứa dataset trên Kaggle.

    Ưu tiên dataset có file bắt đầu bằng E001 vì repo gốc đang dùng
    quy ước này.
    """
    kaggle_input = "/kaggle/input"

    if not os.path.isdir(kaggle_input):
        return None

    # Ưu tiên thư mục chứa E001*.csv.
    for root, _, files in os.walk(kaggle_input):
        if any(f.startswith("E001") and f.lower().endswith(".csv")
               for f in files):
            return root

    # Fallback: thư mục đầu tiên có CSV.
    for root, _, files in os.walk(kaggle_input):
        if any(f.lower().endswith(".csv") for f in files):
            return root

    return None


if __name__ == "__main__":
    KAGGLE_INPUT = auto_find_kaggle_input()

    # Scratch disk: không chiếm quota output /kaggle/working.
    KAGGLE_SCRATCH = "/kaggle/tmp/arcnn/dataset/MyData"

    if KAGGLE_INPUT is None:
        raise FileNotFoundError(
            "Không tìm thấy CSV trong /kaggle/input."
        )

    process_raw_csv_to_train_val_test(
        input_dir=KAGGLE_INPUT,
        output_dir=KAGGLE_SCRATCH,
        seq_len=SEQ_LEN,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=42,
    )