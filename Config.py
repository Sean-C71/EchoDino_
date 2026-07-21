import random
import numpy as np
import torch
import os
import re
import sys
import pydicom
import pydicom.pixel_data_handlers.util as dicomutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append("/data/sc159/EchoDino")

import logging
import time
from datetime import datetime

"""
model_path
"""
DINO_DEFAULT_PATH = "/data/sc159/dinoV3/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

"""
data_path
"""
# TCH_VIEW_FRAME_DATASET_ROOT
TCH_VIEW_FRAME_DATASET_ROOT = "/rdf/data/RDF/forEchoDino"


def get_logger(log_file=None, name=__name__):
    """
    Create and return a logger that outputs to both console and file (if provided).
    Logs include timestamp, logger name, level, and message.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter with timestamp
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (always active)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if log_file is provided)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)  # File gets more detailed logs
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Prevent duplicate logs from higher-level loggers
    logger.propagate = False

    return logger


def seed_everything(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)