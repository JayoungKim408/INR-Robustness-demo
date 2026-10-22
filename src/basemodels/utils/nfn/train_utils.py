import math
import torch
import os
import random
import numpy as np

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # 필요 시 try/except


def get_linear_warmup_with_cos_decay(optimizer, total_steps, warmup_steps, decay_start=None):
    if decay_start is None:
        decay_start = warmup_steps
    def lr_schedule(step):
        if step < warmup_steps:
            # Linear warmup
            lr = min(1., step / warmup_steps)
        else:
            # Cosine decay
            progress = max((step - decay_start) / (total_steps - decay_start), 0)
            lr = 0.5 * (1 + math.cos(math.pi * progress))
        return lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)


def get_linear_warmup(optimizer, total_steps, warmup_steps):
    del total_steps
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1., i / warmup_steps))