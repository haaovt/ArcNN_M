import torch
import torch.nn as nn
import numpy as  np
import tqdm
import json
import os

from utils.model_utils import MyModel, MyModel2


models_dict = {
    "MyModel"   :   MyModel,
    "MyModel2"  :   MyModel2,
}

class BaseTrainer():
    """
    A Trainer is responsible for: handling training steps, overall training, save/load checkpoints, save results and logging
    """
    # This is just an reference for code structure, most of the code should be modified according to application

    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.model = models_dict[cfgs['model']] # some implementation requires seperate models for feature extractor and classifier, change if needed 

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])
        
        if cfgs['loss_type'] == "CrossEntropy": 
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")
        
        if self.cuda:
            self.model.cuda()


    def train_step(self, train_loader):
        """
        Perform one train step (one epoch step) by iterating over all train_loader. Single batch step require reimplementation.
        """
        self.model.train()
        loader_len = 0.0
        total_loss_class = 0.0

        for batch_idx, (all_x, all_y) in enumerate(train_loader):
            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()

            loss_class = self.loss_type(self.predict(all_x), all_y)

            self.optimizer.zero_grad()
            loss_class.backward()
            self.optimizer.step()

            total_loss_class += loss_class.item()
            loader_len += all_x.shape[0]

        total_loss_class /= loader_len

        return {'loss_class' : total_loss_class} # return a dict in cases where many losses is calculated, else, return total_loss_class is enough


    def predict(self, x):
        return self.model(x)


    def validate_step(self, loader):
        """
        Perform validation step over the entire split (can be train, val or test split) 
        """
        self.model.eval()
        acc = 0.0
        loader_len = 0.0

        pred_list = []

        for batch_idx, minibatch in enumerate(loader):
            all_x = minibatch.batch_feature
            all_y = minibatch.batch_label

            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()

            with torch.no_grad():
                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                num_corrects = torch.eq(pred, all_y).sum()
                pred_list.extend(zip(pred.cpu().numpy(),all_y.cpu().numpy())) # save predictions if needed

                acc += num_corrects.cpu().numpy()      
                loader_len += all_x.shape[0]

        self.model.train()

        return pred_list, acc/loader_len


    def train(self, num_epochs, train_loader, val_loader, test_loader, ckpt_freq=10, results_dir=None, cur_epoch=0):
        """
        Trainer function that performs training over (num_epochs-cur_epoch) epochs.        
        """

        loss_list = [] 

        iterator = tqdm(range(cur_epoch, num_epochs), total=num_epochs-cur_epoch, unit='epoch', position=0, leave=True)
        for epoch in iterator:
            loss_list.append(self.train_step(train_loader))

            if (epoch+1)%ckpt_freq == 0: # change if needed
                _, train_acc = self.validate_step(train_loader)
                _, val_acc = self.validate_step(val_loader)
                _, test_acc = self.validate_step(test_loader)

                loss_list[-1].update({'train_acc': train_acc,
                                    'val_acc': val_acc,
                                    'test_acc': test_acc,
                                    'epoch': float(epoch+1)})

                for key in loss_list[-1].keys():
                    tqdm.write(f"{key}".ljust(15), end = "")
                tqdm.write("")

                for key in loss_list[-1].keys():
                    tqdm.write(f"{loss_list[-1][key]:.10f}".ljust(15), end="")
                tqdm.write("")

                self.save_ckpt(epoch, results_dir)

                # Save the best model, implement if needed
                # if val_acc > best_score:
                #     best_score = val_acc
                #     self.save_ckpt(epoch, results_dir, is_best=True)
        
        output_file = open(os.path.join(results_dir, 'loss_list'), 'a', encoding='utf-8')
        for dic in loss_list:
            json.dump(dic, output_file)
            output_file.write("\n")
        
        return loss_list


    def save_ckpt(self, epoch, results_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Best_ckpt.pth.rar')
        else:
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




















