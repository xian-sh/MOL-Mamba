#!/usr/bin/python
# -*- coding:utf-8 -*-

# From https://github.com/THUNLP-MT/PS-VAE

import os
import sys
import torch
import numpy as np
import random
import logging
import pandas as pd

def setup_logger(name, save_dir, filename="training_log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False

    return logger


LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR']
LEVELS_MAP = None


def init_map():
    global LEVELS_MAP, LEVELS
    LEVELS_MAP = {}
    for idx, level in enumerate(LEVELS):
        LEVELS_MAP[level] = idx


def get_prio(level):
    global LEVELS_MAP
    if LEVELS_MAP is None:
        init_map()
    return LEVELS_MAP[level.upper()]


def print_log(s, level='INFO', end='\n', no_prefix=False):
    pth_prio = get_prio(os.getenv('LOG', 'INFO'))
    prio = get_prio(level)
    if prio >= pth_prio:
        if not no_prefix:
            print(level.upper() + '::', end='')
        print(s, end=end)
        sys.stdout.flush()


def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()



def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)