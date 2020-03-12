import logging
import os
import random

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def init_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), mode='w')
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(log_format)
    fh.setFormatter(log_format)
    logging.basicConfig(level=logging.INFO, handlers=[ch, fh])
    tf.get_logger().setLevel(logging.ERROR)
