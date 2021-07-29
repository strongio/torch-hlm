from typing import Sequence, Iterator

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


def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    assert log_diag.shape[:-1] == off_diag.shape[:-1]

    rank = log_diag.shape[-1]
    L1 = torch.diag_embed(torch.exp(log_diag))

    L2 = torch.zeros_like(L1)
    mask = torch.tril_indices(rank, rank, offset=-1)
    L2[mask[0], mask[1]] = off_diag
    return L1 + L2


def chunk_grouped_data(*tensors, group_ids: Sequence):
    """
    XXX
    :param tensors:
    :param group_ids:
    :return:
    """
    group_ids = validate_1d_like(np.asanyarray(group_ids))

    # torch.split requires we put groups into contiguous chunks:
    sort_idx = np.argsort(group_ids)
    group_ids = group_ids[sort_idx]
    tensors = [x[sort_idx] for x in tensors]

    # much faster approach to chunking than something like `[X[gid==group_ids] for gid in np.unique(group_ids)]`:
    _, counts_per_group = np.unique(group_ids, return_counts=True)
    counts_per_group = counts_per_group.tolist()

    group_data = []
    for chunk_tensors in zip(*(torch.split(x, counts_per_group) for x in tensors)):
        group_data.append(chunk_tensors)
    return group_data


def validate_tensors(*args) -> Iterator:
    for arg in args:
        # arg = torch.as_tensor(arg)
        if torch.isnan(arg).any():
            raise ValueError("`nans` in tensor")
        if torch.isinf(arg).any():
            raise ValueError("`infs` in tensor")
        yield arg


def validate_group_ids(group_ids: Sequence, num_grouping_factors: int) -> np.ndarray:
    group_ids = np.asanyarray(group_ids)
    if num_grouping_factors > 1:
        if len(group_ids.shape) != 2 or len(group_ids.shape[1]) != num_grouping_factors:
            raise ValueError(
                f"There are {num_grouping_factors} grouping-factors, so `group_ids` should be 2d with 2nd "
                f"dimension of this extent."
            )
    else:
        group_ids = validate_1d_like(group_ids)[:, None]
    return group_ids


def get_yhat_r(design: dict,
               X: torch.Tensor,
               group_ids: np.ndarray,
               res_per_gf: dict) -> torch.Tensor:
    """
    Get yhat for random-effects.

    :param design: A dictionary with keys as grouping factors and values as indices in the model-matrix.
    :param X: The model-matrix.
    :param group_ids: The group-ids.
    :param res_per_gf: A dictionary with keys as grouping factors and values as random-effect coefficients.
    :return: A tensor with rows corresponding to the model-matrix and columns corresponding to the grouping-factors.
    """
    yhat_r = torch.empty(*group_ids.shape)
    for i, (gf, col_idx) in enumerate(design.items()):
        Xr = torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1)
        _, group_idx = np.unique(group_ids[:, i], return_inverse=True)
        betas_broad = res_per_gf[gf][group_idx]
        yhat_r[:, i] = (Xr * betas_broad).sum(1)
    return yhat_r


def get_to_kwargs(x) -> dict:
    if isinstance(x, torch.nn.Module):
        return get_to_kwargs(next(iter(x.parameters())))
    return {'dtype': x.dtype, 'device': x.device}
