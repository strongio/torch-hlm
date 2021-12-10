import math

from typing import Optional

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import lazy_property

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property


class WoodburyMultivariateNormal(Distribution):
    """
    sigma_diag + Z re_precision^{-1} Z'
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "Z": constraints.independent(constraints.real, 2),
        "sigma_diag": constraints.independent(constraints.positive, 1),
        "re_precision": constraints.positive_definite
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self,
                 loc: torch.Tensor,
                 sigma_diag: torch.Tensor,
                 Z: torch.Tensor,
                 re_precision: torch.Tensor,
                 validate_args: Optional[bool] = None):

        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]

        if Z.dim() < 2:
            raise ValueError("``Z`` must be at least two-dimensional, with optional leading batch dimensions")
        if Z.shape[-2:-1] != event_shape:
            raise ValueError("``Z`` must be a batch of matrices with shape {} x m".format(event_shape[0]))
        if sigma_diag.shape[-1:] != event_shape:
            raise ValueError("``sigma_diag`` must be a batch of vectors with shape {}".format(event_shape))
        # todo: validate re_precision

        loc_ = loc.unsqueeze(-1)
        sigma_diag_ = sigma_diag.unsqueeze(-1)
        try:
            loc_, self.Z, sigma_diag_ = torch.broadcast_tensors(loc_, Z, sigma_diag_)
        except RuntimeError as e:
            raise ValueError(
                "Incompatible batch shapes: loc {}, Z {}, sigma_diag {}".format(
                    loc.shape, Z.shape, sigma_diag.shape
                )
            ) from e
        self.loc = loc_[..., 0]
        self.sigma_diag = sigma_diag_[..., 0]  # A^-1
        self.re_precision = re_precision  # B
        batch_shape = self.loc.shape[:-1]
        # TODO: broadcast re_precision to batch_shape

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @cached_property
    def O(self) -> torch.Tensor:
        return self.re_precision + (self.sigma_diag.unsqueeze(-1) * self.Z).permute(0, 2, 1) @ self.Z

    # @functools.cached_property
    # def A(self) -> torch.Tensor:
    #     return torch.diag_embed(1 / self.sigma_diag)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        import pdb
        pdb.set_trace()
        diff = value - self.loc
        manhalob = (diff * self._lr_solve(diff)).sum(1)
        n = len(value)
        return n * torch.log(torch.tensor(2 * math.pi)) + self._logdet() + manhalob

    def _lr_solve(self, x) -> torch.Tensor:
        A_x = torch.diag_embed((1 / self.sigma_diag * x))
        return A_x - 1 / self.sigma_diag * self.Z @ torch.linalg.solve(self.O, (self.Z @ A_x))

    def _logdet(self) -> torch.Tensor:
        A = torch.diag_embed(1 / self.sigma_diag)  # TODO
        return torch.linalg.slogdet(self.O)[1] - \
               torch.linalg.slogdet(A)[1] - \
               torch.linalg.slogdet(self.re_precision)[1]

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("TODO")
        # new = self._get_checked_instance(WoodburyMultivariateNormal, _instance)
        # batch_shape = torch.Size(batch_shape)
        # loc_shape = batch_shape + self.event_shape
        # new.loc = self.loc.expand(loc_shape)
        # new.cov_diag = self.cov_diag.expand(loc_shape)
        # new.cov_factor = self.cov_factor.expand(loc_shape + self.cov_factor.shape[-1:])
        # new.cov_factor_inner = self.cov_factor_inner.expand(loc_shape + self.cov_factor_inner.shape[-1:])
        # super().__init__(batch_shape,
        #                                                self.event_shape,
        #                                                validate_args=False)
        # new._validate_args = self._validate_args
        # return new

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        raise NotImplementedError("TODO")

    @lazy_property
    def covariance_matrix(self):
        raise NotImplementedError("TODO")

    @lazy_property
    def precision_matrix(self):
        raise NotImplementedError("TODO")

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError("TODO")
