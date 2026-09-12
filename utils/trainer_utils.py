import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import json
import os

from utils.model_utils import ArcNN, SHLNN

models_dict = {
    "ArcNN": ArcNN,
    "SHLNN": SHLNN,
}

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class BaseTrainer():
    def __init__(self, cfgs, args):
        self.cuda = args.cuda
        self.model = models_dict[cfgs['model']](cfgs)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])
        
        if cfgs['loss_type'] == "CrossEntropy": 
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")
        
        if self.cuda:
            self.model.cuda()

    def train_step(self, train_loader):
        self.model.train()
        loader_len = 0
        total_loss = 0.0
        acc_corrects = 0

        # Đã gỡ bỏ tqdm ở đây để không in rác ra màn hình
        for all_x, all_y in train_loader:
            if self.cuda:
                all_x, all_y = all_x.cuda(), all_y.cuda()

            pred = self.predict(all_x)
            loss = self.loss_type(pred, all_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred_classes = pred.max(1)
            acc_corrects += torch.eq(pred_classes, all_y).sum().item()
            
            total_loss += loss.item() * all_x.shape[0]
            loader_len += all_x.shape[0]

        if loader_len == 0: return {'loss_class': 0.0, 'train_acc': 0.0}
        return {'loss_class': total_loss / loader_len, 'train_acc': acc_corrects / loader_len}

    def predict(self, x):
        return self.model(x)

    def validate_step(self, loader):
        self.model.eval()
        loader_len = 0
        total_loss = 0.0
        acc_corrects = 0

        # Đã gỡ bỏ tqdm ở đây
        for all_x, all_y in loader:
            if self.cuda:
                all_x, all_y = all_x.cuda(), all_y.cuda()

            with torch.no_grad():
                pred = self.predict(all_x)
                loss = self.loss_type(pred, all_y)
                total_loss += loss.item() * all_x.shape[0]

                _, pred_classes = pred.max(1)
                acc_corrects += torch.eq(pred_classes, all_y).sum().item()
                loader_len += all_x.shape[0]

        self.model.train()

        if loader_len == 0: return 0.0, 0.0
        return acc_corrects / loader_len, total_loss / loader_len

    def train(self, num_epochs, train_loader, val_loader, test_loader, ckpt_freq=10, results_dir=None, cur_epoch=0):
        loss_list = [] 
        early_stopping = EarlyStopping(patience=3)

        # Chỉ giữ lại 1 thanh tqdm duy nhất đếm tổng số Epoch
        iterator = tqdm(range(cur_epoch, num_epochs), total=num_epochs-cur_epoch, unit='epoch', desc="Epochs")
        for epoch in iterator:
            train_metrics = self.train_step(train_loader)
            val_acc, val_loss = self.validate_step(val_loader)
            test_acc, test_loss = self.validate_step(test_loader)
            
            loss_class = train_metrics['loss_class']
            train_acc = train_metrics['train_acc']

            epoch_stats = {
                'epoch': float(epoch + 1),
                'loss_class': loss_class,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': test_acc
            }
            loss_list.append(epoch_stats)

            # In ra các cột thông số đúng theo định dạng bạn yêu cầu
            tqdm.write(f"Epoch {epoch+1:02d}/{num_epochs} | loss_class: {loss_class:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | val_loss: {val_loss:.4f} | test_acc: {test_acc:.4f}")

            if (epoch + 1) % ckpt_freq == 0:
                self.save_ckpt(epoch, results_dir)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered. Stopping training.")
                break

        with open(os.path.join(results_dir, 'loss_list.json'), 'w', encoding='utf-8') as output_file:
            json.dump(loss_list, output_file, indent=4)
        
        return loss_list
    
    def extract_features(self, loader, save_path):
        self.model.eval()
        total_samples = len(loader.dataset)
        
        features_arr = None
        labels_arr = np.empty((total_samples,), dtype=np.int64)

        tqdm.write(f"Extracting features to {save_path}...")
        current_idx = 0
        
        with torch.no_grad():
            for all_x, all_y in loader:
                if self.cuda:
                    all_x = all_x.cuda()

                feature = self.model(all_x, extract_features=True).cpu().numpy()
                batch_size = feature.shape[0]
                
                if features_arr is None:
                    feature_dim = feature.shape[1]
                    features_arr = np.empty((total_samples, feature_dim), dtype=np.float32)

                features_arr[current_idx : current_idx + batch_size] = feature
                labels_arr[current_idx : current_idx + batch_size] = all_y.numpy()
                current_idx += batch_size

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        label_path = save_path.replace('features', 'labels')
        
        np.save(label_path, labels_arr)
        np.save(save_path, features_arr)
        tqdm.write("Feature extraction completed.")

    def save_ckpt(self, epoch, results_dir, is_best=False):
        checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Epoch_{epoch}_ckpt.pth.rar')
        state_dict = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['torch_rng'])
        np.random.set_state(state_dict['np_random'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        return epoch