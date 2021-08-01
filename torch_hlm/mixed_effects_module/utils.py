from typing import Sequence, Iterator, Dict

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


def validate_tensors(*args: torch.Tensor) -> Iterator[torch.Tensor]:
    for arg in args:
        if torch.isnan(arg).any():
            raise ValueError("`nans` in tensor")
        if torch.isinf(arg).any():
            raise ValueError("`infs` in tensor")
        yield arg


def validate_group_ids(group_ids: Sequence, num_grouping_factors: int) -> np.ndarray:
    group_ids = np.asanyarray(group_ids)
    if num_grouping_factors > 1:
        if len(group_ids.shape) != 2 or group_ids.shape[1] != num_grouping_factors:
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


def pad_res_per_gf(res_per_gf: Dict[str, torch.Tensor],
                   group_ids1: Sequence,
                   group_ids2: Sequence,
                   verbose: bool = False) -> Dict[str, torch.Tensor]:
    # there is no requirement that all groups in `group_ids` are present in `group_data`, or vice versa, so
    # need to map the re_solve output
    res_per_gf_padded = {}
    for gf_i, gf in enumerate(res_per_gf):
        ugroups_target = {gid: i for i, gid in enumerate(np.unique(group_ids1[:, gf_i]))}
        ugroups_solve = {gid: i for i, gid in enumerate(np.unique(group_ids2[:, gf_i]))}
        set1 = set(ugroups_solve) - set(ugroups_target)
        if set1 and verbose:
            print(f"there are {len(set1):,} groups in `re_solve_data` but not in `X`")
        set2 = set(ugroups_target) - set(ugroups_solve)
        if set2 and verbose:
            print(f"there are {len(set2):,} groups in `X` but not in `re_solve_data`")

        res_per_gf_padded[gf] = torch.zeros(
            (len(ugroups_target), res_per_gf[gf].shape[-1]), device=res_per_gf[gf].device, dtype=res_per_gf[gf].dtype
        )
        for gid_target, idx_target in ugroups_target.items():
            idx_solve = ugroups_solve.get(gid_target)
            if idx_solve is None:
                continue
            res_per_gf_padded[gf][idx_target] = res_per_gf[gf][idx_solve]
    return res_per_gf_padded
