import pandas as pd
import numpy as np
import os
import random
from scipy.signal import butter, lfilter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def butter_bandpass_filter(data, lowcut=90000, highcut=110000, fs=250000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return lfilter(b, a, data)

def process_single_file(args):
    target_fold, file_path, output_dir, seq_len = args
    fold_dir = os.path.join(output_dir, target_fold)
    os.makedirs(fold_dir, exist_ok=True)
    
    df = pd.read_csv(file_path)
    current = df['Current_A'].values
    voltage = df['Voltage_V'].values
    base_name = os.path.basename(file_path).replace('.csv', '').replace('.CSV', '')
    
    samples_list = []
    targets_list = []
    
    for i in range(0, len(current) - seq_len + 1, seq_len):
        chunk_c = current[i : i + seq_len]
        chunk_v = voltage[i : i + seq_len]
        label = 1 if (15 <= np.mean(chunk_v) <= 20) else 0 
            
        filtered_c = butter_bandpass_filter(chunk_c)
        c_min, c_max = np.min(filtered_c), np.max(filtered_c)
        norm_c = (filtered_c - c_min) / (c_max - c_min) if c_max > c_min else filtered_c - c_min
            
        samples_list.append(norm_c.reshape(1, seq_len).astype(np.float32))
        targets_list.append(label)
        
    if samples_list:
        samples_arr = np.stack(samples_list) 
        targets_arr = np.array(targets_list, dtype=np.int64) 
        
        # Tách làm 2 file riêng biệt để dùng mmap_mode='r'
        np.save(os.path.join(fold_dir, f"{base_name}_x.npy"), samples_arr)
        np.save(os.path.join(fold_dir, f"{base_name}_y.npy"), targets_arr)
        
    return target_fold

def process_raw_csv_to_kfold(input_dir, output_dir, seq_len=512, k_folds=5):
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if not csv_files: return
        
    random.seed(42)
    random.shuffle(csv_files)
    
    tasks = []
    for idx, path in enumerate(csv_files):
        fold_number = (idx % k_folds) + 1
        tasks.append((f"Fold_{fold_number}", path, output_dir, seq_len))
        
    print(f"Bắt đầu chia {len(csv_files)} file vào {k_folds} nhóm K-Fold...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing K-Fold"))

def auto_find_kaggle_input():
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            if file.startswith('E001'): return root
    return None

if __name__ == '__main__':
    KAGGLE_INPUT = auto_find_kaggle_input()
    if KAGGLE_INPUT:
        KAGGLE_OUTPUT = "/kaggle/working/dataset/MyData"
        process_raw_csv_to_kfold(input_dir=KAGGLE_INPUT, output_dir=KAGGLE_OUTPUT, k_folds=5)