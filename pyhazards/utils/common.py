import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logging.basicConfig(level=os.getenv("PYHAZARD_LOGLEVEL", "INFO"))
    return logging.getLogger(name or "pyhazards")
