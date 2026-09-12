import numpy as np
import os
from sklearn.decomposition import PCA
import joblib

def run_pca_compression(data_dir='./dataset_pca', target_evr=0.80):
    print(f"Running PCA compression targeting EVR={target_evr*100}% in {data_dir}...")

    x_train = np.load(os.path.join(data_dir, 'train_features.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    
    # Bổ sung đọc tập Validation
    x_val = np.load(os.path.join(data_dir, 'val_features.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))

    x_test = np.load(os.path.join(data_dir, 'test_features.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))

    pca = PCA(n_components=target_evr)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)
    
    n_components_found = pca.n_components_
    print(f"Optimal dimensions found: {n_components_found}")

    out_dir = os.path.join(data_dir, f'compressed_{n_components_found}d')
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, 'train_features_pca.npy'), x_train_pca)
    np.save(os.path.join(out_dir, 'train_labels.npy'), y_train)
    np.save(os.path.join(out_dir, 'val_features_pca.npy'), x_val_pca)
    np.save(os.path.join(out_dir, 'val_labels.npy'), y_val)
    np.save(os.path.join(out_dir, 'test_features_pca.npy'), x_test_pca)
    np.save(os.path.join(out_dir, 'test_labels.npy'), y_test)

    joblib.dump(pca, os.path.join(out_dir, 'pca_model.joblib'))
    print("Compressed features and PCA model saved successfully.")

if __name__ == '__main__':
    # Chạy kịch bản này độc lập để tạo lại file nén cho tất cả các fold
    import glob
    fold_dirs = glob.glob('./dataset_pca/fold_*')
    for fold_dir in fold_dirs:
        run_pca_compression(fold_dir, target_evr=0.80)