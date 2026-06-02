from tqdm import tqdm
from utils.cmd_parser import get_agrs_parser
from initialize import init_train, init_test, init_trainer
import json
import os

def main():
    # cfgs are contains in ./configs/config.yaml
    # args are arguments passed in terminal command, find details abour args in cmd_parser.py
    # run 'python train.py -c ./configs/config.yaml train --no_cuda' to run with CPU 
    cfgs, args = get_agrs_parser()

    if args.mode == 'train':
        trainer, loaders, results_dir = init_train(cfgs, args)

        if cfgs['load_checkpoint'] == 'None':
            cur_epoch = 0
        else:
            cur_epoch = trainer.load_ckpt(cfgs['load_checkpoint'])
        
        for i_loader, (train_loader, val_loader, test_loader) in enumerate(loaders):
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

if __name__ == '__main__':
    main()




