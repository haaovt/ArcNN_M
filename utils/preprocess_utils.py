"""
preprocess_utils.py
-------------------
Tiền xử lý dữ liệu cho ArcNN.

Mục tiêu:
1. Đọc Current_A và Voltage_V từ CSV.
2. Giữ nguyên cách chia dữ liệu theo FILE trước khi tạo K-fold.
   -> Các window của cùng một CSV không bị rơi vào nhiều fold.
3. Tạo window dài 512 điểm.
4. Gán nhãn dựa trên điện áp trung bình 15-20 V.
5. Lọc dòng điện trong dải 90-110 kHz.
6. Min-Max về [0, 1].
7. Lưu X/Y theo từng file để CustomDataset có thể dùng mmap_mode='r'.

LƯU Ý VỀ PAPER:
Paper mô tả CBF (Composite Bandpass Filter) phần cứng với dải 90-110 kHz.
Code Kaggle hiện tại không có mô hình CBF phần cứng, nên Butterworth
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
    cần thiết khi dữ liệu đầu vào của Kaggle là raw CSV.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def normalize_minmax(signal):
    """Đưa một sample về khoảng [0, 1]."""
    signal = np.asarray(signal, dtype=np.float32)

    s_min = np.min(signal)
    s_max = np.max(signal)

    if s_max > s_min:
        return (signal - s_min) / (s_max - s_min)

    # Trường hợp tín hiệu phẳng.
    return np.zeros_like(signal, dtype=np.float32)


def make_samples_from_csv(file_path, seq_len=SEQ_LEN):
    """
    Đọc một CSV và tạo toàn bộ sample 512 điểm.

    Label chỉ dùng Voltage_V để tạo ground-truth:
        15 V <= mean(V) <= 20 V -> arc = 1
        ngược lại                 -> normal = 0

    Voltage không được đưa vào ArcNN.
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

    # Không overlap: mỗi window có đúng 512 điểm.
    for start in range(0, len(current) - seq_len + 1, seq_len):
        chunk_current = current[start:start + seq_len]
        chunk_voltage = voltage[start:start + seq_len]

        # Label theo tiêu chí điện áp trong paper.
        #
        # KHÔNG dùng mean(voltage) vì arc thường chỉ xảy ra trong
        # một phần của window — mean kéo voltage trung bình xuống
        # và mislabel arc window thành normal.
        #
        # Ví dụ: arc 100 điểm (17V) + normal 412 điểm (0V)
        #   mean = 3.3V → labeled NORMAL (sai)
        #   max  = 17V  → labeled ARC    (đúng)
        #
        # Paper (Section II.B): dùng steady-state voltage 15-20V
        # để nhận biết arc. Khi arc đang cháy, voltage luôn >= 15V.
        # Nên nếu MAX voltage trong window >= 15V → có arc xảy ra.
        max_voltage = float(np.max(np.abs(chunk_voltage)))
        label = int(max_voltage >= ARC_VOLTAGE_MIN)

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


def process_single_file(args):
    """
    Worker xử lý một CSV.

    Mỗi CSV được lưu thành:
        Fold_X/<filename>_x.npy
        Fold_X/<filename>_y.npy

    Cách này tương thích với CustomDataset hiện tại.
    """
    target_fold, file_path, output_dir, seq_len = args

    fold_dir = os.path.join(output_dir, target_fold)
    os.makedirs(fold_dir, exist_ok=True)

    x, y = make_samples_from_csv(file_path, seq_len=seq_len)

    if x is None:
        return target_fold, os.path.basename(file_path), 0

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    np.save(os.path.join(fold_dir, f"{base_name}_x.npy"), x)
    np.save(os.path.join(fold_dir, f"{base_name}_y.npy"), y)

    return target_fold, base_name, len(y)


def process_raw_csv_to_kfold(
    input_dir,
    output_dir,
    seq_len=SEQ_LEN,
    k_folds=5,
    seed=42,
    num_workers=None,
):
    """
    Chia CSV thành K-fold rồi mới preprocessing.

    QUAN TRỌNG:
    - K-fold được chia ở mức FILE, không chia ở mức window.
    - Điều này tránh leakage giữa các window của cùng một thí nghiệm.
    - Giữ K=5 theo yêu cầu của project.

    output_dir nên nằm ở scratch disk của Kaggle:
        /kaggle/tmp/arcnn/dataset/MyData

    Không nên dùng /kaggle/working cho toàn bộ dataset trung gian.
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

    tasks = []
    for idx, file_path in enumerate(csv_files):
        fold_number = (idx % k_folds) + 1
        tasks.append(
            (
                f"Fold_{fold_number}",
                file_path,
                output_dir,
                seq_len,
            )
        )

    print(
        f"Chia {len(csv_files)} CSV vào {k_folds} folds "
        f"(seed={seed})..."
    )

    # Nếu không chỉ định, ProcessPoolExecutor tự chọn số worker.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_file, tasks),
                total=len(tasks),
                desc="Preprocessing K-Fold",
            )
        )

    total_samples = sum(r[2] for r in results)

    print("\n===== PREPROCESSING DONE =====")
    print(f"CSV files : {len(csv_files)}")
    print(f"K-fold    : {k_folds}")
    print(f"Samples   : {total_samples}")
    print(f"Output    : {output_dir}")

    return results


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

    process_raw_csv_to_kfold(
        input_dir=KAGGLE_INPUT,
        output_dir=KAGGLE_SCRATCH,
        seq_len=SEQ_LEN,
        k_folds=5,
        seed=42,
    )