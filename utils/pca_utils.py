"""
pca_utils.py
------------
PCA fit trên TRAIN, transform VAL/TEST — đúng Eq.(5) và mục III.B
của paper (giữ 80% explained variance ratio).

Không còn K-fold: chỉ có một thư mục input/output duy nhất, không
cần đặt tên thư mục theo số chiều PCA tìm được.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
import joblib


def run_pca_compression(data_dir, out_dir=None, target_evr=0.80):
    """
    Đọc train/val/test_features.npy + *_labels.npy trong data_dir,
    fit PCA trên TRAIN (target_evr = EVR mong muốn, paper dùng 0.80),
    transform VAL/TEST, rồi lưu kết quả vào out_dir (mặc định = data_dir).

    Trả về số chiều PCA thực tế tìm được (int).
    """
    if out_dir is None:
        out_dir = data_dir

    print(f"Running PCA compression targeting EVR={target_evr*100:.0f}% in {data_dir}...")

    x_train = np.load(os.path.join(data_dir, 'train_features.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))

    x_val = np.load(os.path.join(data_dir, 'val_features.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))

    x_test = np.load(os.path.join(data_dir, 'test_features.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))

    # PCA chỉ fit trên TRAIN — val/test chỉ transform, không leak.
    pca = PCA(n_components=target_evr)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)

    n_components_found = int(pca.n_components_)
    print(f"Optimal dimensions found: {n_components_found} (paper báo cáo 70 dimensions)")

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'train_features_pca.npy'), x_train_pca)
    np.save(os.path.join(out_dir, 'train_labels.npy'), y_train)
    np.save(os.path.join(out_dir, 'val_features_pca.npy'), x_val_pca)
    np.save(os.path.join(out_dir, 'val_labels.npy'), y_val)
    np.save(os.path.join(out_dir, 'test_features_pca.npy'), x_test_pca)
    np.save(os.path.join(out_dir, 'test_labels.npy'), y_test)

    joblib.dump(pca, os.path.join(out_dir, 'pca_model.joblib'))
    print("Compressed features and PCA model saved successfully.")

    return n_components_found


if __name__ == '__main__':
    # Chạy độc lập trên một thư mục feature duy nhất (không K-fold).
    run_pca_compression(data_dir='./dataset_pca', target_evr=0.80)