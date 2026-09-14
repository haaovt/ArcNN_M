"""
preprocess_utils.py
-------------------
Tiền xử lý dữ liệu cho ArcNN.

Mục tiêu:
1. Đọc Current_A và Voltage_V từ CSV.
2. Chia dữ liệu theo FILE thành đúng 3 tập Train/Val/Test MỘT LẦN DUY
   NHẤT (KHÔNG dùng K-fold) — giống cách bài báo chia dữ liệu ở Bảng II.
   -> Các window của cùng một CSV không bị rơi vào nhiều tập khác nhau.
3. Tạo window dài 512 điểm.
4. Gán nhãn dựa trên điện áp trong khoảng ổn định 15-20 V (paper mục II.B).
5. Lọc dòng điện trong dải 90-110 kHz.
6. Min-Max về [0, 1].
7. Lưu X/Y theo từng file để CustomDataset có thể dùng mmap_mode='r'.

LƯU Ý VỀ PAPER:
Paper mô tả CBF (Composite Bandpass Filter) phần cứng với dải 90-110 kHz.
Code hiện tại không có mô hình CBF phần cứng, nên Butterworth digital
band-pass dưới đây chỉ là phép xấp xỉ phần lọc 90-110 kHz. Không nên gọi
đây là CBF phần cứng chính xác.

VỀ TỈ LỆ TRAIN/VAL/TEST:
Paper không nêu tỉ lệ chia trực tiếp, nhưng Bảng II cho số mẫu thực tế:
    Training  = 15957
    Test      = 4981
    Validation= 3985
    Total     = 24918
=> xấp xỉ Train 64% / Validation 16% / Test 20%. Mặc định dưới đây dùng
đúng tỉ lệ này; có thể chỉnh qua tham số nếu cần.
"""

import os
import random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm


FS = 250_000          # Sampling frequency: 250 kHz
LOWCUT = 90_000       # Lower cutoff: 90 kHz
HIGHCUT = 110_000     # Upper cutoff: 110 kHz
FILTER_ORDER = 4
SEQ_LEN = 512         # Paper sử dụng 512 điểm / sample
ARC_VOLTAGE_MIN = 15
ARC_VOLTAGE_MAX = 20

#train/val/test ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def butter_bandpass_filter(
    data,
    lowcut=LOWCUT,
    highcut=HIGHCUT,
    fs=FS,
    order=FILTER_ORDER,
):
    
    nyq = 0.5 * fs  #Nyquist frequency
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
        có ít nhất 1 điểm với 15V <= |V| <= 20V trong window -> arc = 1
        ngược lại                                             -> normal = 0

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

    # Không overlap: mỗi window có đúng 512 điểm.
    for start in range(0, len(current) - seq_len + 1, seq_len):
        chunk_current = current[start:start + seq_len]
        chunk_voltage = voltage[start:start + seq_len]

        # Label theo tiêu chí điện áp trong paper (mục II.B): điện áp
        # hồ quang ổn định nằm trong khoảng ĐÓNG 15-20V. Chỉ cần có ít
        # nhất 1 điểm trong window rơi vào khoảng này -> có arc.
        abs_voltage = np.abs(chunk_voltage)
        in_arc_range = (abs_voltage >= ARC_VOLTAGE_MIN) & (abs_voltage <= ARC_VOLTAGE_MAX)
        label = int(np.any(in_arc_range))

        #Bandpass filter và normalize trước khi lưu sample.
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
        Train/<filename>_x.npy , Train/<filename>_y.npy
    hoặc Val/... , hoặc Test/... tùy vào split được gán.

    Cách này tương thích với CustomDataset hiện tại (CustomDataset đọc
    tất cả *_x.npy trong một thư mục).
    """
    target_split, file_path, output_dir, seq_len = args

    split_dir = os.path.join(output_dir, target_split)
    os.makedirs(split_dir, exist_ok=True)

    x, y = make_samples_from_csv(file_path, seq_len=seq_len)

    if x is None:
        return target_split, os.path.basename(file_path), 0

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    np.save(os.path.join(split_dir, f"{base_name}_x.npy"), x)
    np.save(os.path.join(split_dir, f"{base_name}_y.npy"), y)

    return target_split, base_name, len(y)


def process_raw_csv_to_split(
    input_dir,
    output_dir,
    seq_len=SEQ_LEN,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=42,
    num_workers=None,
):
   
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio phải bằng 1.0"
    )

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
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    if n_val == 0 or n_test == 0:
        print(
            f"CẢNH BÁO: chỉ có {n} file CSV nên Val/Test có thể bị "
            f"rỗng (Train={n_train}, Val={n_val}, Test={n_test}). "
            f"Cần thêm file CSV thô để có tập Val/Test thực sự đánh "
            f"giá được model."
        )

    split_assignment = (
        ["Train"] * n_train + ["Val"] * n_val + ["Test"] * n_test
    )

    tasks = [
        (split_assignment[idx], file_path, output_dir, seq_len)
        for idx, file_path in enumerate(csv_files)
    ]

    print(
        f"Chia {n} CSV -> Train={n_train} / Val={n_val} / Test={n_test} "
        f"(seed={seed})..."
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_file, tasks),
                total=len(tasks),
                desc="Preprocessing Train/Val/Test",
            )
        )

    total_samples = sum(r[2] for r in results)

    print("\n===== PREPROCESSING DONE =====")
    print(f"CSV files : {n}")
    print(f"Split     : Train={n_train} | Val={n_val} | Test={n_test}")
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

    process_raw_csv_to_split(
        input_dir=KAGGLE_INPUT,
        output_dir=KAGGLE_SCRATCH,
        seq_len=SEQ_LEN,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42,
    )