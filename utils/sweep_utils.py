import numpy as np
import copy

def get_random_search_configs(cfgs, seed, search, model):
    new_cfgs = copy.deepcopy(cfgs)
    new_cfgs['train_id'] = f'seed{seed}_search{search}_{model}'
    new_cfgs['model'] = model

    # SỬA (đã đối chiếu với file paper thật, Section IV-A):
    # "Both ArcNN and SHLNN are trained on batches of size 64" — paper
    # dùng batch_size=64 CỐ ĐỊNH cho cả 2 model, không sweep. Bản cũ
    # random-search batch_size trong [32,64,128] cho MỌI model, tức là
    # phần lớn lần sweep (2/3) dùng batch_size khác paper. Nếu bạn vẫn
    # muốn dò batch_size cho dataset của riêng mình, sửa dòng dưới lại
    # thành random.choice — nhưng để khớp đúng paper thì cố định 64.
    new_cfgs['batch_size'] = 64

    new_cfgs['weight_decay'] = float(np.random.choice([1e-4, 1e-5]))

    # SỬA (learning rate không thực sự theo model — bug khớp comment):
    # Comment gốc ở đây đã ghi đúng ý định "ưu tiên các cụm xung quanh
    # 0.0002 (ArcNN) và 0.001 (SHLNN)" — đúng theo paper Section IV-A
    # ("ArcNN is trained with a learning rate of 0.0002 and SHLNN of
    # 0.001"). Nhưng code THỰC TẾ lại dùng chung 1 danh sách
    # [0.0001, 0.0002, 0.0005, 0.001] cho CẢ HAI model, không hề tách
    # theo `model` — nghĩa là ArcNN có thể bị sweep ra learning_rate=
    # 0.001 (5x lớn hơn giá trị paper dùng) và SHLNN có thể bị sweep ra
    # 0.0001 (10x nhỏ hơn giá trị paper). Giờ tách danh sách theo đúng
    # từng model, mỗi danh sách xoay quanh giá trị paper dùng.
    if model == 'ArcNN':
        new_cfgs['learning_rate'] = float(
            np.random.choice([0.0001, 0.0002, 0.0003, 0.0005])
        )
    else:  # SHLNN
        new_cfgs['learning_rate'] = float(
            np.random.choice([0.0005, 0.001, 0.002, 0.003])
        )

    # Tìm kiếm tham số đặc thù cho các mạng trong bài báo
    if new_cfgs['model'] == 'ArcNN':
        new_cfgs['ArcNN'] = {}
        new_cfgs['ArcNN']['num_inputs'] = 1
        new_cfgs['ArcNN']['dropout'] = float(np.random.choice([0.1, 0.2, 0.3]))

    elif new_cfgs['model'] == 'SHLNN':
        new_cfgs['SHLNN'] = {}
        new_cfgs['SHLNN']['num_inputs'] = 70  # Hoặc tuỳ biến theo số dimensions tìm ra từ thuật toán PCA trên
        new_cfgs['SHLNN']['hidden_neurons'] = int(np.random.choice([90, 110, 130]))

    return new_cfgs
