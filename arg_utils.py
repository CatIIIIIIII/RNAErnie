"""
This module defines some common functions for argument parsing.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/7 7:41 PM
"""

import argparse
import random
import os.path as osp

import numpy as np

import paddle
from paddlenlp.utils.log import logger


def set_seed(seed):
    """set seed for random, numpy and paddle

    Args:
        seed (int): seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def default_logdir():
    """generate default log dir

    Returns:
        path.Path: path of log dir
    """
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return osp.join("runs", current_time)


def str2bool(v):
    """convert string args to boolean

    Args:
        v (str): from args

    Returns:
        boolean: True or False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v):
    """convert string to list

    Args:
        v (str): from args

    Returns:
        list: sperated by ','
    """
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        vs = v.split(",")
        return [v.strip() for v in vs]
    else:
        raise argparse.ArgumentTypeError("Str value seperated by ', ' expected.")


def str2intlist(v):
    """convert string to int list

    Args:
        v (string): from args

    Returns:
        list: int list
    """
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        vs = v.split(",")
        if vs[-1] == "":
            vs = vs[:-1]
        return [int(v.strip()) for v in vs]
    else:
        raise argparse.ArgumentTypeError("Str value seperated by ', ' expected.")


def list2str(list_value):
    """convert list to string

    Args:
        v (list): int list

    Returns:
        str: s
    """
    if isinstance(list_value, list):
        res = ""
        for x in list_value:
            res = res + x + ","
        return res[:-1]
    else:
        raise NotImplementedError


def print_config(args=None, key=""):
    """print the configuration of the experiment

    Args:
        args (argparse.Namespace): from args

    Returns:
        None
    """
    logger.debug("=" * 60)

    logger.debug('{:^40}'.format("{} Configuration Arguments".format(key)))
    logger.debug('{:30}:{}'.format("paddle commit id", paddle.version.commit))

    for k, v in vars(args).items():
        logger.debug('{:30}:{}'.format(k, v))
