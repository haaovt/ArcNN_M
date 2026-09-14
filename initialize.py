import os
import shutil
from ruamel.yaml import YAML
import torch
import numpy as np
from utils.init_utils import get_trainer, get_dataloader

def init_train(cfgs, args):
    # Set rng seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Create directories
    results_dir = os.path.join('./results', cfgs['dataset'], cfgs['train_id'])
    if cfgs['load_checkpoint'] == 'None':
        if os.path.isdir(results_dir):
            raise ValueError(f'{results_dir} Already existed!')

        os.makedirs(os.path.join(results_dir, 'ckpts'))

        yaml = YAML()
        # SỬA LỖI SYNTAX (nghiêm trọng — file này sẽ không import được):
        # Bản gốc là f'config_{cfgs['train_id']}.yaml' — dùng NHÁY ĐƠN
        # cho cả f-string lẫn key truy cập bên trong {}. Trước Python
        # 3.12 (PEP 701), lồng cùng loại dấu nháy bên trong f-string là
        # SyntaxError, nghĩa là bất kỳ ai import module này trên
        # Python <3.12 sẽ crash ngay khi parse file, chưa cần chạy tới
        # dòng này. Sửa bằng cách đổi dấu nháy bên trong thành nháy kép
        # để an toàn trên mọi phiên bản Python (kể cả Kaggle thường
        # chạy 3.10/3.11).
        config_filename = f'config_{cfgs["train_id"]}.yaml'
        with open(os.path.join(results_dir, config_filename), 'w') as f:
            yaml.dump(cfgs, f)

        # SỬA: bản gốc hardcode './configs/config.yaml' — nếu bạn chạy
        # với config_stage1.yaml hoặc config_stage2.yaml (như trong
        # bộ config bạn gửi), file lưu lại vào results_dir vẫn luôn là
        # config.yaml, SAI với config thực sự đã dùng để train, gây
        # nhầm lẫn khi xem lại kết quả sau này.
        #
        # Mình không có utils/cmd_parser.py nên không chắc `args` có
        # thuộc tính nào giữ đường dẫn config gốc (quy ước phổ biến là
        # `args.config`). Dùng getattr với fallback về hành vi cũ để
        # không crash nếu tên thuộc tính khác — NHƯNG bạn nên kiểm tra
        # lại cmd_parser.py và sửa tên thuộc tính cho khớp nếu cần.
        source_config_path = getattr(args, 'config', './configs/config.yaml')
        if os.path.isfile(source_config_path):
            shutil.copy(source_config_path, results_dir)
        else:
            print(
                f"[CẢNH BÁO] Không tìm thấy config gốc tại "
                f"'{source_config_path}' để copy vào {results_dir}. "
                f"Đã lưu bản dump của cfgs ở '{config_filename}' rồi, "
                f"nhưng bạn nên kiểm tra lại tên thuộc tính config "
                f"trong utils/cmd_parser.py."
            )


    loaders = get_dataloader(cfgs, args)
    trainer = get_trainer(cfgs, args)

    return trainer, loaders, results_dir

def init_trainer(cfgs, args):
    return get_trainer(cfgs, args)


def init_test(cfgs, args):
    """
    Implement this if you want to test on another dataset
    """
    pass
