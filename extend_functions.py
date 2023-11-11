import copy

import numpy as np


def try_to_eliminate_constant(arr):
    sample_a = None
    for a in arr:
        if type(a) == np.ndarray:
            sample_a = a
            break
    if sample_a is None:
        return False
    for i, a in enumerate(arr):
        if type(a) != np.ndarray:
            arr[i] = np.full_like(sample_a, a)
    return True


def avg(*arr):
    arr = list(arr)
    if try_to_eliminate_constant(arr):
        return np.mean(np.stack(arr), axis=0)
    else:
        return np.mean(arr)


def np_max(*arr):
    arr = list(arr)
    if try_to_eliminate_constant(arr):
        return np.max(np.stack(arr), axis=0)
    else:
        return np.max(arr)


def np_min(*arr):
    arr = list(arr)
    if try_to_eliminate_constant(arr):
        return np.min(np.stack(arr), axis=0)
    else:
        return np.min(arr)


def add(*arr):
    sum = arr[0]
    for x in arr[1:]:
        sum = sum + x
    return sum


def sub(*arr):
    sum = arr[0]
    for x in arr[1:]:
        sum = sum - x
    return sum


def protect_divide(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    if type(b) != np.ndarray:
        if type(a) == np.ndarray:
            b = np.full_like(a, b)
        elif np.abs(b) < 1e-6:
            b = 1
    if type(b) == np.ndarray:
        b[np.abs(b) < 1e-6] = 1
    return np.nan_to_num(np.divide(a, b))


def protect_mod(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    if type(b) != np.ndarray:
        if type(a) == np.ndarray:
            b = np.full_like(a, b)
        elif np.abs(b) < 1e-6:
            b = 1
    if type(b) == np.ndarray:
        b[np.abs(b) < 1e-6] = 1
    return np.nan_to_num(np.mod(a, b))


def protect_log(log_fun, a):
    a = copy.deepcopy(a)

    a = np.abs(a)
    if type(a) != np.ndarray:
        if a < 1e-6:
            a = 1
        return np.nan_to_num(log_fun(a))
    a[np.abs(a) < 1e-6] = 1
    return np.nan_to_num(log_fun(a))


def protect_log2(a):
    return protect_log(np.log2, a)


def protect_loge(a):
    return protect_log(np.log, a)


def protect_log10(a):
    return protect_log(np.log10, a)


def protect_log1p(a):
    if type(a) != np.ndarray:
        if a < 0:
            a = 0
        return np.nan_to_num(np.log1p(a))
    a[a < 0] = 0
    return np.nan_to_num(np.log1p(a))


def protect_sqrt(a):
    return np.sqrt(np.abs(a))
