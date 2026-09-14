"""
preprocess_utils.py
-------------------
Tiền xử lý dữ liệu cho ArcNN.

Mục tiêu:
1. Đọc Current_A và Voltage_V từ CSV.
2. Chia dữ liệu theo FILE trước khi tạo K-fold, đồng thời CÂN BẰNG
   nhãn arc/normal giữa các fold (xem assign_files_to_folds_balanced).
   -> Các window của cùng một CSV không bị rơi vào nhiều fold, và
      không fold nào bị lệch hẳn về một nhãn.
3. Tạo window dài 512 điểm.
4. Gán nhãn dựa trên điện áp trung bình 15-20 V.
5. Lọc dòng điện trong dải 90-110 kHz (lọc trên toàn bộ file trước,
   rồi mới cắt window — xem SỬA LỖI THỨ TỰ LỌC trong make_samples_from_csv).
6. Min-Max về [0, 1].
7. Lưu X/Y theo từng file để CustomDataset có thể dùng mmap_mode='r'.

LƯU Ý VỀ PAPER (đã đối chiếu với file paper thật — Yan, Li, Duan,
"A Simplified Current Feature Extraction and Deployment Method for
DC Series Arc Fault Detection", IEEE TIE 2024):
- CBF phần cứng của paper có -3dB cutoff 90-110kHz (Fig.4/Table I),
  khớp đúng LOWCUT/HIGHCUT dưới đây. Nhưng CBF là mạch analog (voltage
  follower + MFBF + Sallen-Key), không phải digital IIR order cụ thể
  nào — nên Butterworth digital band-pass dưới đây (FILTER_ORDER=4)
  vẫn chỉ là phép XẤP XỈ đặc tính dải tần, không phải con số lấy từ
  paper. Không nên gọi đây là CBF phần cứng chính xác.
- FS=250_000 và SEQ_LEN=512 đã ĐƯỢC XÁC NHẬN khớp cả paper ("sampling
  rate of 250 kHz", "input to ArcNN is 512 points per sample") VÀ file
  CSV mẫu bạn gửi (Time_s cách nhau đúng 4µs = 250kHz).
- ARC_VOLTAGE_MIN=15 khớp cận dưới paper ("voltage... always between
  15 and 20V"). ARC_VOLTAGE_MAX=20 là cận trên paper NHƯNG — xem
  make_samples_from_csv() — mình đã kiểm tra thực tế trên file CSV mẫu
  bạn gửi và thấy vùng arc ổn định của file đó nằm ở ~47-54V, KHÔNG
  phải 15-20V (tên file "0050V" khớp với việc điện áp arc ổn định của
  chính file này là ~50V). Nên 15-20V nhiều khả năng là con số riêng
  của rig thí nghiệm trong paper, không phải hằng số áp dụng chung cho
  mọi cấu hình. Áp cận trên 20V cứng vào điều kiện gán nhãn sẽ LÀM SAI
  toàn bộ nhãn của các file có điện áp arc ổn định > 20V (như file mẫu
  này) — do đó KHÔNG áp cận trên, chỉ giữ điều kiện một chiều
  `>= ARC_VOLTAGE_MIN` như bản gốc (xem chi tiết trong hàm dưới).
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
# ARC_VOLTAGE_MAX = 20 là cận trên NÊU TRONG PAPER ("...always between
# 15 and 20V"), nhưng KHÔNG được dùng để chặn nhãn (xem lý do ở
# make_samples_from_csv / phần LƯU Ý VỀ PAPER phía trên) — điện áp arc
# ổn định thực tế trong dữ liệu của bạn có thể cao hơn 20V tuỳ cấu
# hình (file mẫu ~50V). Giữ hằng số này lại chỉ để tham khảo/đối chiếu
# paper, không đưa vào điều kiện gán nhãn.
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

    # SỬA LỖI THỨ TỰ LỌC (quan trọng, ảnh hưởng chất lượng feature):
    #
    # Bản cũ cắt window 512 điểm TRƯỚC, rồi mới gọi
    # butter_bandpass_filter riêng cho từng window (mỗi lần gọi
    # filtfilt là một lần lọc độc lập, tự "mồi" pha đầu-cuối cho riêng
    # 512 điểm đó). Vì đây là band-pass hẹp (90-110kHz, order=4, tương
    # đương ~8 cực đối với bandpass), filtfilt cần một đoạn transient
    # ổn định ở hai đầu tín hiệu; lọc từng đoạn 512 điểm độc lập tạo méo
    # dạng sóng ở CẢ HAI ĐẦU CỦA MỖI WINDOW, tức là méo lặp lại 512
    # điểm/lần trên toàn bộ tín hiệu. Điều này không khớp với CBF phần
    # cứng trong bài báo — CBF lọc LIÊN TỤC trên dòng tín hiệu đang chạy
    # qua, không "khởi động lại" mỗi 512 điểm.
    #
    # Cách sửa: lọc band-pass trên TOÀN BỘ Current_A của cả file trước,
    # sau đó mới cắt thành các window 512 điểm. Nhờ vậy chỉ có 2
    # đầu-cuối của CẢ FILE bị ảnh hưởng bởi transient của bộ lọc, thay
    # vì mọi window đều bị ảnh hưởng ở cả hai đầu.
    filtered_current_full = butter_bandpass_filter(current)

    samples = []
    labels = []

    # Không overlap: mỗi window có đúng 512 điểm.
    for start in range(0, len(current) - seq_len + 1, seq_len):
        chunk_current_filtered = filtered_current_full[start:start + seq_len]
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
        #
        # CHỦ ĐÍCH KHÔNG chặn thêm cận trên (<=20V, tức ARC_VOLTAGE_MAX):
        # đã kiểm tra trên file CSV mẫu bạn gửi, vùng arc ổn định của
        # file đó là ~47-54V (không phải 15-20V như số liệu cụ thể
        # trong paper — có thể do rig/gap khác nhau giữa các thí
        # nghiệm, tên file "0050V" gợi ý điện áp arc thiết kế ~50V cho
        # riêng file này). Nếu thêm điều kiện `<= 20` sẽ mislabel toàn
        # bộ đoạn arc ổn định 47-54V này thành "normal" — sai nặng hơn
        # nhiều so với việc không chặn cận trên. Điều kiện một chiều
        # `>= ARC_VOLTAGE_MIN` vẫn đúng cho mọi mức điện áp arc quan
        # sát được (15V lẫn ~50V), miễn là baseline "normal" luôn gần
        # 0V (đã kiểm tra: baseline thực tế ~0.5-1.7V, cách xa 15V).
        max_voltage = float(np.max(np.abs(chunk_voltage)))
        label = int(max_voltage >= ARC_VOLTAGE_MIN)

        # Min-Max (lọc band-pass đã được thực hiện trên toàn file ở trên).
        normalized_current = normalize_minmax(chunk_current_filtered)

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


def estimate_file_label_counts(file_path, seq_len=SEQ_LEN):
    """
    Ước lượng NHANH số window arc/normal của một file, chỉ dựa vào cột
    Voltage_V (không đọc/lọc Current_A) — dùng để cân bằng K-fold.

    Dùng chung tiêu chí nhãn với make_samples_from_csv (max voltage
    trong window >= ARC_VOLTAGE_MIN) để số liệu nhất quán, nhưng bỏ qua
    hoàn toàn phần band-pass filter (không cần thiết chỉ để đếm nhãn),
    nên rẻ hơn nhiều so với chạy toàn bộ make_samples_from_csv.
    """
    df = pd.read_csv(file_path, usecols=["Voltage_V"])
    voltage = df["Voltage_V"].to_numpy(dtype=np.float32)

    n_windows = len(voltage) // seq_len
    if n_windows == 0:
        return 0, 0

    arc_count = 0
    for start in range(0, n_windows * seq_len, seq_len):
        chunk_voltage = voltage[start:start + seq_len]
        max_voltage = float(np.max(np.abs(chunk_voltage)))
        if max_voltage >= ARC_VOLTAGE_MIN:
            arc_count += 1

    return arc_count, n_windows - arc_count


def assign_files_to_folds_balanced(csv_files, k_folds, seed=42, seq_len=SEQ_LEN):
    """
    Gán từng CSV vào một trong k_folds fold, CHIA THEO FILE và CÂN BẰNG
    nhãn arc/normal giữa các fold.

    SỬA (K-FOLD CHƯA HỢP LÝ):
    Bản cũ: shuffle danh sách file rồi chia round-robin (idx % k_folds).
    Cách này vẫn đảm bảo không rơi window của cùng 1 file vào nhiều
    fold, NHƯNG không quan tâm tỉ lệ arc/normal của từng file. Với dữ
    liệu arc fault (nhãn arc thường là thiểu số và phân bố không đều
    giữa các file thí nghiệm), round-robin ngẫu nhiên dễ tạo ra fold
    lệch nhãn nặng — ví dụ 1 fold gần như toàn "normal". Hậu quả:
      - Khi fold đó là test-fold: kết quả đánh giá không phản ánh đúng
        khả năng phát hiện arc thực sự.
      - Khi fold đó nằm trong tập train: class weight tính trên các
        fold train (trainer_utils.py._compute_class_weights) dễ bị lệch
        cực đoan, thậm chí gặp lớp vắng mặt hoàn toàn (chia cho 0) —
        đây là lý do 2 vấn đề K-fold và ClassWeight bạn nêu có liên
        quan trực tiếp với nhau.

    Cách sửa: ước lượng nhanh số window arc/normal của từng file
    (estimate_file_label_counts — chỉ đọc cột Voltage_V nên rẻ), sau đó
    dùng thuật toán tham lam "largest file first": xếp file có nhiều
    sample nhất trước, mỗi lần gán 1 file vào fold đang "thiếu hụt" cả
    về TỔNG số sample lẫn số sample ARC so với mức lý tưởng (tổng chia
    đều cho k_folds), tính bằng một điểm số chuẩn hoá:

        score(fold) = fold_total[fold] / ideal_total
                    + fold_arc[fold]   / ideal_arc

    (mỗi số hạng ~1.0 nghĩa là fold đó đã nhận đúng phần lý tưởng của
    riêng tiêu chí đó; chọn fold có score THẤP NHẤT để gán file tiếp
    theo vào). Chuẩn hoá về cùng thang đo giúp cả hai tiêu chí (kích
    thước & tỉ lệ arc) đều có trọng số tương đương khi cộng lại — nếu
    chỉ ưu tiên riêng arc-count như (fold_arc, fold_total) kiểu so
    sánh từ-điển (lexicographic) sẽ khiến 1 fold có arc thấp hút hết
    các file "normal" bất kể kích thước đã phình to cỡ nào.

    Lưu ý: đây vẫn là heuristic tham lam (không tối ưu toàn cục), nên
    với dataset có RẤT ÍT file (ví dụ chỉ vài file cho 5 fold) hoặc
    phân bố nhãn cực kỳ lưỡng cực giữa các file, kết quả có thể chưa
    hoàn hảo — nhưng với số file thực tế của một dataset arc-fault
    (hàng chục file trở lên) heuristic này cho kết quả cân bằng tốt.

    Bước ước lượng nhãn đọc thêm 1 lượt Voltage_V cho mỗi file trước
    khi preprocessing thật — chi phí này thường nhỏ vì chỉ đọc 1 cột
    và không chạy band-pass filter.
    """
    random.seed(seed)

    file_stats = []
    for file_path in tqdm(csv_files, desc="Ước lượng nhãn theo file (để cân bằng fold)"):
        arc_count, normal_count = estimate_file_label_counts(file_path, seq_len)
        file_stats.append((file_path, arc_count, normal_count))

    # Xáo trộn trước khi sort để không thiên vị theo tên file khi nhiều
    # file có cùng số lượng sample (tie-break ngẫu nhiên, có seed).
    random.shuffle(file_stats)

    # "Largest first": xếp file nhiều sample trước giúp thuật toán tham
    # lam cân bằng tốt hơn so với xếp file theo thứ tự ngẫu nhiên/tên.
    file_stats.sort(key=lambda item: item[1] + item[2], reverse=True)

    total_samples_all = sum(a + n for _, a, n in file_stats)
    total_arc_all = sum(a for _, a, n in file_stats)
    ideal_total = total_samples_all / k_folds if k_folds > 0 else 1
    ideal_arc = total_arc_all / k_folds if total_arc_all > 0 else 0

    fold_arc = [0] * k_folds
    fold_total = [0] * k_folds
    fold_assignment = {f"Fold_{i + 1}": [] for i in range(k_folds)}

    def _fold_score(i):
        score = fold_total[i] / ideal_total if ideal_total > 0 else 0.0
        if ideal_arc > 0:
            score += fold_arc[i] / ideal_arc
        return score

    for file_path, arc_count, normal_count in file_stats:
        total = arc_count + normal_count

        # Chọn fold có score THẤP NHẤT (đang "thiếu" nhiều nhất so với
        # phần lý tưởng của nó, tính chung cả size lẫn arc-count).
        fold_idx = min(range(k_folds), key=_fold_score)

        fold_name = f"Fold_{fold_idx + 1}"
        fold_assignment[fold_name].append(file_path)
        fold_arc[fold_idx] += arc_count
        fold_total[fold_idx] += total

    print("Phân bố sample (arc / tổng) theo fold sau khi cân bằng:")
    for i in range(k_folds):
        arc_ratio = (fold_arc[i] / fold_total[i] * 100) if fold_total[i] > 0 else 0.0
        print(f"  Fold_{i + 1}: arc={fold_arc[i]:>7} / total={fold_total[i]:>7}  ({arc_ratio:.1f}% arc)")

    return fold_assignment


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
    - Việc gán file vào fold còn được CÂN BẰNG theo nhãn arc/normal
      (xem assign_files_to_folds_balanced), không chỉ round-robin ngẫu
      nhiên như trước — tránh fold lệch nhãn nặng.
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

    print(
        f"Chia {len(csv_files)} CSV vào {k_folds} folds theo FILE, "
        f"cân bằng nhãn arc/normal (seed={seed})..."
    )

    # SỬA: thay shuffle + round-robin bằng gán cân bằng theo nhãn
    # (xem docstring assign_files_to_folds_balanced).
    fold_assignment = assign_files_to_folds_balanced(
        csv_files, k_folds, seed=seed, seq_len=seq_len
    )

    tasks = []
    for fold_name, files_in_fold in fold_assignment.items():
        for file_path in files_in_fold:
            tasks.append((fold_name, file_path, output_dir, seq_len))

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
