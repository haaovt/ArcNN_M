from tqdm import tqdm
from utils.cmd_parser import get_agrs_parser
from initialize import init_train, init_test, init_trainer
import json
import os
from utils.pca_utils import run_pca_compression

def main():
    cfgs, args = get_agrs_parser()
    
    # cfgs are contains in ./configs/config.yaml
    # args are arguments passed in terminal command, find details abour args in cmd_parser.py
    # run 'python train.py -c ./configs/config.yaml train --no_cuda' to run with CPU 

    if args.mode == 'train':
        trainer, loaders, results_dir = init_train(cfgs, args)

        if cfgs['load_checkpoint'] == 'None':
            cur_epoch = 0
        else:
            cur_epoch = trainer.load_ckpt(cfgs['load_checkpoint'])
        
        for i_loader, (train_loader, val_loader, test_loader) in enumerate(loaders):
            
            # --- CƠ CHẾ TỰ ĐỘNG ĐỌC SỐ CHIỀU PCA ---
            if cfgs['model'] == 'SHLNN':
                # Bốc thử 1 mẫu dữ liệu đầu tiên từ tập train hiện tại
                sample_x, _ = train_loader.dataset[0]
                # Đo kích thước ma trận đặc trưng (số chiều PCA của fold này)
                dynamic_num_inputs = sample_x.shape[0] 
                
                # Tự động ghi đè thông số num_inputs vào cấu hình ảo
                cfgs[cfgs['model']]['num_inputs'] = dynamic_num_inputs
                print(f"[{'='*40}]")
                print(f"Fold {i_loader}: Auto-configured SHLNN input size -> {dynamic_num_inputs} dimensions")
            # ---------------------------------------

            # Khi khởi tạo trainer, SHLNN sẽ tự động đọc số num_inputs vừa được cập nhật ở trên
            trainer = init_trainer(cfgs, args)
            num_epochs = cfgs['num_epochs']
            dom_results_dir = os.path.join(results_dir, f'test_dom_{i_loader}')
            if not os.path.isdir(dom_results_dir):
                os.makedirs(os.path.join(dom_results_dir,'ckpts'), exist_ok=True)

            loss_list = trainer.train(cur_epoch=cur_epoch, 
                                num_epochs=num_epochs, 
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                test_loader=test_loader,
                                results_dir=dom_results_dir,
                                ckpt_freq=cfgs['ckpt_freq'])
            
            if cfgs['model'] == 'ArcNN':
                print(f'Extracting features for PCA (Fold {i_loader})...')
                pca_data_dir = f'./dataset_pca/fold_{i_loader}'
                os.makedirs(pca_data_dir, exist_ok=True)

                trainer.extract_features(train_loader, save_path=os.path.join(pca_data_dir, 'train_features.npy'))
                trainer.extract_features(val_loader, save_path=os.path.join(pca_data_dir, 'val_features.npy'))
                trainer.extract_features(test_loader, save_path=os.path.join(pca_data_dir, 'test_features.npy'))
                
                print('Feature extraction completed. Running PCA...')
                run_pca_compression(data_dir=pca_data_dir, target_evr=0.80)

if __name__ == '__main__':
    main()