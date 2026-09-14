import os
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.trainer_utils import BaseTrainer
from utils.dataset_utils import CustomDataset, PCADataset

trainers_dict = {'BaseTrainer': BaseTrainer}
datasets_dict = {'CustomDataset': CustomDataset, 'MyData': CustomDataset, 'PCADataset': PCADataset}

def get_trainer(cfgs, args):
    return trainers_dict[cfgs['trainer']](cfgs, args)

def get_dataloader(cfgs, args):
    if cfgs['dataset'] == 'PCADataset':
        return get_pca_dataloader_folds(cfgs, args)
    return get_dataloader_folds(cfgs, args)

def get_pca_dataloader_folds(cfgs, args):
    loaders = []
    dataset_dir = cfgs['rootdir'] 
    # SỬA: os.listdir() không đảm bảo thứ tự -> sort để Fold_1..Fold_5
    # (hay round_1..round_5 phía train.py) luôn ứng với đúng thư mục,
    # không phụ thuộc OS/filesystem (xem chi tiết ở get_dataloader_folds).
    fold_list = sorted(os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f)))
    for fold_dir in fold_list:
        comp_dirs = [d for d in os.listdir(fold_dir) if d.startswith('compressed')]
        if not comp_dirs: continue
        comp_dir = os.path.join(fold_dir, comp_dirs[0])
        train_dataset = PCADataset(os.path.join(comp_dir, 'train_features_pca.npy'), os.path.join(comp_dir, 'train_labels.npy'))
        val_dataset = PCADataset(os.path.join(comp_dir, 'val_features_pca.npy'), os.path.join(comp_dir, 'val_labels.npy'))
        test_dataset = PCADataset(os.path.join(comp_dir, 'test_features_pca.npy'), os.path.join(comp_dir, 'test_labels.npy'))
        train_loaders = DataLoader(dataset=train_dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
        val_loaders = DataLoader(dataset=val_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
        test_loaders = DataLoader(dataset=test_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
        loaders.append((train_loaders, val_loaders, test_loaders))
    return loaders


def _split_fold_files(fold_dir, ratios, seed=None):
    """
    Chia các file *_x.npy trong MỘT thư mục fold thành N phần theo
    tỉ lệ `ratios`, CHIA THEO FILE (không theo window).

    SỬA LỖI RÒ RỈ DỮ LIỆU (train/val, và train/val/test khi chỉ có 1 fold):
    Bản cũ tạo một CustomDataset cho cả fold_dir (gộp mọi window của
    mọi file CSV trong fold thành một danh sách liên tục), rồi
    `np.random.shuffle` trên WINDOW-INDEX và Subset theo tỉ lệ. Vì
    thao tác shuffle/cắt này xảy ra SAU KHI các file đã bị gộp phẳng,
    hai window liền kề của CÙNG MỘT file CSV (rất tương quan — cùng
    một arc event, cùng nền nhiễu của lần đo đó) hoàn toàn có thể vừa
    rơi vào train vừa rơi vào val/test. Đây là rò rỉ dữ liệu, khiến
    val loss/acc dùng để chọn best checkpoint bị lạc quan giả tạo, và
    đi ngược lại chính nguyên tắc "không chia 1 file vào nhiều tập"
    mà preprocess_utils.py cố gắng đảm bảo khi tạo Fold_1..Fold_5.

    Cách sửa: liệt kê danh sách file trong fold (CustomDataset.list_file_pairs,
    không load dữ liệu), shuffle DANH SÁCH FILE, cắt theo tỉ lệ `ratios`
    theo SỐ FILE, rồi build một CustomDataset riêng cho mỗi phần từ danh
    sách file tương ứng. Một file luôn thuộc trọn vẹn một phần.

    ratios: tuple tỉ lệ, ví dụ (0.8, 0.2) hoặc (0.7, 0.2, 0.1). Không
        cần tổng đúng bằng 1, hàm tự chuẩn hoá.
    seed: random seed để tái lập kết quả.

    Trả về: list các CustomDataset, cùng độ dài với `ratios`.
    """
    file_pairs = CustomDataset.list_file_pairs(fold_dir)
    n_files = len(file_pairs)

    if n_files == 0:
        raise ValueError(f"Không tìm thấy file *_x.npy nào trong {fold_dir}")

    ratios = np.asarray(ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()
    n_parts = len(ratios)

    if n_files < n_parts:
        # Không đủ file để mỗi phần có tối thiểu 1 file riêng — đây là
        # dấu hiệu dataset/fold quá nhỏ so với số phần cần chia. Vẫn
        # chia được nhưng CẢNH BÁO rõ vì một vài phần buộc phải trống
        # hoặc phải fallback về window-level (chấp nhận rò rỉ nhẹ) để
        # không crash toàn bộ pipeline.
        print(
            f"[CẢNH BÁO] {fold_dir} chỉ có {n_files} file nhưng cần chia "
            f"thành {n_parts} phần theo tỉ lệ {tuple(ratios.tolist())}. "
            f"Không đủ file để chia theo FILE cho từng phần -> dùng tạm "
            f"window-level split cho fold này (có thể còn rò rỉ nhẹ giữa "
            f"các phần). Hãy cân nhắc tăng số file trong fold này."
        )
        dataset = CustomDataset(file_pairs=file_pairs)
        window_idx = np.arange(len(dataset))
        rng = np.random.RandomState(seed)
        rng.shuffle(window_idx)

        counts = np.floor(ratios * len(window_idx)).astype(int)
        counts[-1] = len(window_idx) - counts[:-1].sum()  # phần cuối nhận phần dư
        splits, cursor = [], 0
        for c in counts:
            splits.append(Subset(dataset, window_idx[cursor:cursor + c]))
            cursor += c
        return splits

    rng = np.random.RandomState(seed)
    file_order = rng.permutation(n_files)

    # Số file cho mỗi phần, làm tròn xuống rồi phân bổ phần dư cho các
    # phần có phần thập phân lớn nhất, đảm bảo mỗi phần có >=1 file.
    raw_counts = ratios * n_files
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 1)
    # Điều chỉnh nếu tổng vượt quá n_files do bước np.maximum ở trên.
    while counts.sum() > n_files:
        counts[int(np.argmax(counts))] -= 1
    remainder = n_files - counts.sum()
    fractional = raw_counts - np.floor(raw_counts)
    for i in np.argsort(-fractional)[:remainder]:
        counts[i] += 1

    datasets = []
    cursor = 0
    for part_count in counts:
        files_in_part = [file_pairs[i] for i in file_order[cursor:cursor + int(part_count)]]
        cursor += int(part_count)
        datasets.append(CustomDataset(file_pairs=files_in_part))

    return datasets


def get_dataloader_folds(cfgs, args):
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
    # SỬA (phát hiện khi đọc lại train.py): os.listdir() trả về thứ tự
    # tuỳ hệ điều hành/filesystem, KHÔNG đảm bảo Fold_1, Fold_2, ...
    # Vì train.py lặp `for fold_idx, fold_loaders in enumerate(loaders,
    # start=1)` rồi in "FOLD {fold_idx}" / lưu vào "round_{fold_idx}",
    # nếu fold_list không sort thì "FOLD 1" trong log có thể thực ra
    # đang test trên Fold_3 (hay bất kỳ fold nào) tuỳ lần chạy — gây
    # khó reproduce và log gây hiểu nhầm. Thêm sorted() để đảm bảo thứ
    # tự luôn là Fold_1, Fold_2, ... Fold_5 khớp với tên thư mục.
    # (Lưu ý: sort dạng chuỗi nên chỉ đúng thứ tự số khi <=9 fold —
    # đủ dùng vì project cố định K=5 — nếu sau này tăng K>=10 cần đổi
    # sang natural sort để tránh "Fold_10" đứng trước "Fold_2".)
    subfolders = sorted(os.path.join(dataset_dir, fold) for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, fold)))
    fold_list = subfolders if len(subfolders) > 0 else [dataset_dir]
    loaders = []

    seed = cfgs.get('seed', 42)
    test_fold_cfg = cfgs.get('test_fold', 'None')
    if test_fold_cfg != 'None':    
        target_test_fold = os.path.join(dataset_dir, test_fold_cfg)
        if target_test_fold not in fold_list: raise ValueError(f"Test folder {target_test_fold} not found!")     
        
        train_dataset, val_dataset = [], []
        test_loaders = None
    
        for fold in fold_list:
            if fold == target_test_fold:
                dataset = datasets_dict[cfgs['dataset']](fold, cfgs=cfgs)
                test_loaders = DataLoader(dataset=dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
            else:
                # SỬA: chia theo FILE thay vì theo window-index (xem
                # docstring của _split_fold_files).
                fold_train_ds, fold_val_ds = _split_fold_files(fold, ratios=(0.8, 0.2), seed=seed)
                train_dataset.append(fold_train_ds)
                val_dataset.append(fold_val_ds)
        
        train_loaders = DataLoader(dataset=ConcatDataset(train_dataset), batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
        val_loaders = DataLoader(dataset=ConcatDataset(val_dataset), batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
        loaders.append((train_loaders, val_loaders, test_loaders))
    else:
        if len(fold_list) == 1:
            fold = fold_list[0]

            # SỬA LỖI NGHIÊM TRỌNG:
            # Bản cũ: train_end = 0.7N, val_end = 0.2N (KHÔNG cộng dồn
            # với train_end), rồi val_dataset = idx[train_end:val_end]
            # = idx[0.7N : 0.2N]. Vì 0.2N < 0.7N nên slice này LUÔN
            # RỖNG -> val_dataset trống hoàn toàn. Đồng thời
            # test_dataset = idx[val_end:] = idx[0.2N:] chồng lấn tới
            # ~50% dữ liệu với train_dataset = idx[:0.7N] (cả hai đều
            # chứa các index từ 0.2N đến 0.7N) -> rò rỉ train/test rất
            # nặng, độ chính xác test bị đánh giá sai lệch nghiêm trọng.
            #
            # Sửa: val_end phải CỘNG DỒN (train_end + 0.2N), và chia
            # theo FILE (không theo window-index) để tránh rò rỉ giữa
            # 3 tập, giống các nhánh K-fold khác ở trên.
            train_dataset, val_dataset, test_dataset = _split_fold_files(
                fold, ratios=(0.7, 0.2, 0.1), seed=seed
            )

            train_loaders = DataLoader(dataset=train_dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
            val_loaders = DataLoader(dataset=val_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
            test_loaders = DataLoader(dataset=test_dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
            loaders.append((train_loaders, val_loaders, test_loaders))
        else:
            for current_test_fold in fold_list:
                train_dataset, val_dataset = [], []
                test_loaders = None

                for fold in fold_list:
                    if fold == current_test_fold:
                        dataset = datasets_dict[cfgs['dataset']](fold, cfgs=cfgs)
                        test_loaders = DataLoader(dataset=dataset, batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
                    else:
                        # SỬA: chia theo FILE thay vì theo window-index.
                        fold_train_ds, fold_val_ds = _split_fold_files(fold, ratios=(0.8, 0.2), seed=seed)
                        train_dataset.append(fold_train_ds)
                        val_dataset.append(fold_val_ds)
                
                train_loaders = DataLoader(dataset=ConcatDataset(train_dataset), batch_size=cfgs['batch_size'], shuffle=True, num_workers=args.num_workers)
                val_loaders = DataLoader(dataset=ConcatDataset(val_dataset), batch_size=cfgs['batch_size'], shuffle=False, num_workers=args.num_workers)
                loaders.append((train_loaders, val_loaders, test_loaders))
    return loaders
