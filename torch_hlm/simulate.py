from typing import Optional, Tuple

import numpy as np
import pandas as pd

from torch_hlm.mixed_effects_module.utils import log_chol_to_chol, ndarray_to_tensor


def simulate_raneffects(num_groups: int,
                        obs_per_group: int,
                        num_raneffects: int,
                        std_multi: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param num_groups: Number of groups
    :param obs_per_group: Number of observations per group
    :param num_raneffects: A number > 0 of random-effects, with the first being the intercept.
    :param std_multi: Multiplier for the standard-deviation of the random effects. Defaults to greater variance for the
    intercept than the predictors.
    :return: Two dataframes: the first with predictors, group-indicator, and a (noiseless) `y` response; the second
    with the true random-effects values for each group.
    """
    # generate model-mat
    assert num_raneffects > 0
    X = np.random.multivariate_normal(
        mean=np.zeros(num_raneffects - 1),
        cov=_random_corr_mat(num_raneffects - 1, eps=np.sqrt(num_raneffects)) / np.sqrt(num_raneffects),
        size=num_groups * obs_per_group
    )

    # generate random-effects:
    if num_raneffects < 10:
        rf_cov = _random_cov_mat(num_raneffects, eps=0.1)
    else:
        lr = np.random.randn(num_raneffects, int(np.sqrt(num_raneffects)))
        rf_cov = lr @ lr.T + .1 * np.eye(num_raneffects)
    if std_multi is None:
        std_multi = [0.5] + [0.1] * (num_raneffects - 1)
    if isinstance(std_multi, (int, float)):
        std_multi = [std_multi] * num_raneffects
    std_multi = np.diag(std_multi)
    rf_cov = std_multi @ rf_cov @ std_multi
    raneffects = np.random.multivariate_normal(mean=np.zeros(num_raneffects), cov=rf_cov, size=num_groups)
    df_raneffects = pd.DataFrame(raneffects, columns=[f'x{i}' for i in range(num_raneffects)])
    df_raneffects['group'] = list(range(num_groups))

    # broadcast, generate y (without noise)
    group_idx = []
    for group_id in range(num_groups):
        group_idx.extend([group_id] * obs_per_group)
    group_idx = np.asanyarray(group_idx)
    Xi = np.concatenate([np.ones((len(X), 1)), X], 1)  # intercept
    y_actual = (raneffects[group_idx] * Xi).sum(1)

    # create data-frame
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, num_raneffects)])
    df['y'] = y_actual
    df['group'] = group_idx
    return df, df_raneffects


def _random_cov_mat(rank: int, eps: float = .001) -> np.ndarray:
    L_log_diag = np.random.randn(rank)
    L_off_diag = np.random.randn(int(rank * (rank - 1) / 2))
    L = log_chol_to_chol(ndarray_to_tensor(L_log_diag), ndarray_to_tensor(L_off_diag)).numpy()
    return (L @ L.T) + eps * np.eye(rank)


def _random_corr_mat(rank: int, eps: float = .001) -> np.ndarray:
    cov = _random_cov_mat(rank, eps=eps)
    _norm = np.diag(1 / np.sqrt(np.diag(cov)))
    return _norm @ cov @ _norm
