import json
import os

import numpy as np
import torch


class BaseExp:
    def __init__(self, args):
        cfg = vars(args)
        for key in cfg.keys():
            setattr(self, key, cfg[key])

        self.exp_path = f'{self.static_path}/{self.expid}'
        os.makedirs(self.exp_path, exist_ok=True)

        with open(f'{self.exp_path}/cfg.json', 'w') as f:
            f.write(json.dumps(cfg, indent=4))

        self.set_seed(self.seed)
        self.cfg = cfg

    def set_seed(self, seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def train(self):
        raise
