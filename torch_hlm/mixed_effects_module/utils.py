from typing import Union

import torch
import numpy as np


def validate_1d_like(x):
    if len(x.shape) > 1:
        extra_shape = list(x.shape[1:])
        if extra_shape == ([1] * len(extra_shape)):
            for _ in range(len(extra_shape)):
                x = x.squeeze(-1)
    if len(x.shape) != 1:
        raise ValueError(f"Expected 1D tensor; instead got `{x.shape}`")
    return x


def ndarray_to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f"`{type(x)}` should have been a tensor or ndarray")
    return x


def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    assert log_diag.shape[:-1] == off_diag.shape[:-1]

    rank = log_diag.shape[-1]
    L = torch.diag_embed(torch.exp(log_diag))

    idx = 0
    for i in range(rank):
        for j in range(i):
            L[..., i, j] = off_diag[..., idx]
            idx += 1
    return L
