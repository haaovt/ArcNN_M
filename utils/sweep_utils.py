import numpy as np
import copy 

def get_random_search_configs(cfgs, seed, search, model):
    new_cfgs = copy.deepcopy(cfgs)
    new_cfgs['train_id'] = f'seed{seed}_search{search}_{model}'
    new_cfgs['model'] = model
    
    # Thiết lập learning rate ưu tiên các cụm xung quanh 0.0002 (ArcNN) và 0.001 (SHLNN)
    new_cfgs['learning_rate'] = float(np.random.choice([0.0001, 0.0002, 0.0005, 0.001]))
    new_cfgs['weight_decay'] = float(np.random.choice([1e-4, 1e-5]))
    new_cfgs['batch_size'] = int(np.random.choice([32, 64, 128]))

    # Tìm kiếm tham số đặc thù cho các mạng trong bài báo
    if new_cfgs['model'] == 'ArcNN':
        new_cfgs['ArcNN'] = {}
        new_cfgs['ArcNN']['num_inputs'] = 1
        new_cfgs['ArcNN']['dropout'] = float(np.random.choice([0.1, 0.2, 0.3]))
    
    elif new_cfgs['model'] == 'SHLNN':
        new_cfgs['SHLNN'] = {}
        new_cfgs['SHLNN']['num_inputs'] = 70 # Hoặc tuỳ biến theo số dimensions tìm ra từ thuật toán PCA trên
        new_cfgs['SHLNN']['hidden_neurons'] = int(np.random.choice([90, 110, 130]))

    return new_cfgs