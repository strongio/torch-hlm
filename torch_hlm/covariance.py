"""
"log_cholesky", parameterizes the random-effects covariance using the cholesky factorization (which is itself split
into two tensors: the log-transformed diagonal elements and the off-diagonal). The other supported option is "low_rank",
 which parameterizes the covariance with two tensors: (a) the log-transformed std-devations, and (b) a 'low rank' G*K
tensor where G is the number of random-effects and K is int(sqrt(G)). Then the covariance is D + V @ V.t() where D is a
diagonal-matrix with the std-deviations**2, and V is the low-rank tensor.
"""

import math
from typing import Optional

import torch

from torch_hlm.mixed_effects_module.utils import log_chol_to_chol


class Covariance(torch.nn.Module):
    _nm2cls = {}

    def __init_subclass__(cls, **kwargs):
        cls._nm2cls[cls.alias] = cls

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> 'Covariance':
        klass = cls._nm2cls.get(name)
        if klass is None:
            raise TypeError(f"Unrecognized `name`, options: {set(cls._nm2cls)}")
        return klass(*args, **kwargs)

    def __init__(self, rank: int):
        super(Covariance, self).__init__()
        self.rank = rank

    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    def set(self, tens: torch.Tensor):
        raise NotImplementedError


class LogCholeskyCov(Covariance):
    alias = 'log_cholesky'

    def __init__(self, rank: int):
        super().__init__(rank=rank)
        _num_upper_tri = int(self.rank * (self.rank - 1) / 2)
        self.cholesky_log_diag = torch.nn.Parameter(data=.1 * torch.randn(self.rank))
        self.cholesky_off_diag = torch.nn.Parameter(data=.1 * torch.randn(_num_upper_tri))

    def forward(self) -> torch.Tensor:
        L = log_chol_to_chol(log_diag=self.cholesky_log_diag, off_diag=self.cholesky_off_diag)
        return L @ L.t()

    def set(self, tens: torch.Tensor):
        L = torch.cholesky(tens)
        with torch.no_grad():
            self.cholesky_log_diag[:] = L.diag().log()
            self.cholesky_off_diag[:] = L[tuple(torch.tril_indices(len(L), len(L), offset=-1))]
        assert torch.isclose(self(), tens, atol=1e-04).all()


class LowRankCov(Covariance):
    alias = 'low_rank'

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> 'Covariance':
        if name.startswith('low_rank'):
            low_rank = name.replace('low_rank', '')
            if low_rank.isdigit():
                kwargs['low_rank'] = int(low_rank)
        return super().from_name(name='low_rank', *args, **kwargs)

    def __init__(self, rank: int, low_rank: Optional[int] = None):
        super().__init__(rank=rank)
        if low_rank is None:
            low_rank = int(math.sqrt(self.rank))
        self.lr = torch.nn.Parameter(data=.1 * torch.randn(self.rank, low_rank))
        self.log_std_devs = torch.nn.Parameter(data=.1 * torch.randn(self.rank))

    def forward(self) -> torch.Tensor:
        return self.lr @ self.lr.t() + torch.diag_embed(self.log_std_devs.exp() ** 2)

    def set(self, tens: torch.Tensor):
        if not torch.allclose(tens, torch.eye(len(tens)) * tens):
            raise RuntimeError(f"{type(self).__name__} unable to set non-diagonal cov")
        with torch.no_grad():
            self.log_std_devs[:] = tens.diag().sqrt().log()
            # TODO: zero-out lr?
        # assert torch.isclose(self(), tens, atol=1e-04).all()
