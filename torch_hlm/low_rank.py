from typing import Optional

import torch
from torch import Tensor
from torch.distributions import LowRankMultivariateNormal as TorchLowRankMultivariateNormal, Distribution, constraints
from torch.distributions.utils import lazy_property


class LowRankMultivariateNormal(Distribution):
    r"""
        Creates a multivariate normal distribution with covariance matrix having a low-rank form
        parameterized by :attr:`cov_factor`, :attr:`cov_diag`, and :attr:`cov_factor_inner`::

            covariance_matrix = cov_diag + cov_factor @ cov_factor_inner @ cov_factor.T

    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.independent(constraints.positive, 1),
        "cov_factor_inner": constraints.positive_definite
    }
    support = constraints.real_vector
    has_rsample = False

    def __init__(self,
                 loc: Tensor,
                 cov_factor: Tensor,
                 cov_diag: Tensor,
                 cov_factor_inner: Optional[Tensor] = None,
                 validate_args: Optional[bool] = None):

        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]

        if cov_factor.dim() < 2:
            raise ValueError("cov_factor must be at least two-dimensional, "
                             "with optional leading batch dimensions")
        if cov_factor.shape[-2:-1] != event_shape:
            raise ValueError("cov_factor must be a batch of matrices with shape {} x m"
                             .format(event_shape[0]))
        if cov_diag.shape[-1:] != event_shape:
            raise ValueError("cov_diag must be a batch of vectors with shape {}".format(event_shape))
        if cov_factor_inner is None:
            raise NotImplementedError("TODO: identity matrix")
        else:
            pass  # TODO: validate

        loc_ = loc.unsqueeze(-1)
        cov_diag_ = cov_diag.unsqueeze(-1)
        try:
            loc_, self.cov_factor, cov_diag_, self.cov_factor_inner = torch.broadcast_tensors(
                loc_, cov_factor, cov_diag_, cov_factor_inner
            )
        except RuntimeError as e:
            raise ValueError("Incompatible batch shapes: loc {}, cov_factor {}, cov_diag {}, cov_factor_inner_ {}"
                             .format(loc.shape, cov_factor.shape, cov_diag.shape, self.cov_factor_inner.shape)) from e
        self.loc = loc_[..., 0]
        self.cov_diag = cov_diag_[..., 0]
        batch_shape = self.loc.shape[:-1]

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LowRankMultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new.cov_diag = self.cov_diag.expand(loc_shape)
        new.cov_factor = self.cov_factor.expand(loc_shape + self.cov_factor.shape[-1:])
        new.cov_factor_inner = self.cov_factor_inner.expand(loc_shape + self.cov_factor_inner.shape[-1:])
        super(LowRankMultivariateNormal, new).__init__(batch_shape,
                                                       self.event_shape,
                                                       validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        raise NotImplementedError("TODO")

    @lazy_property
    def covariance_matrix(self):
        raise NotImplementedError("TODO")
        # covariance_matrix = (torch.matmul(self._unbroadcasted_cov_factor,
        #                                   self._unbroadcasted_cov_factor.transpose(-1, -2))
        #                      + torch.diag_embed(self._unbroadcasted_cov_diag))
        # return covariance_matrix.expand(self._batch_shape + self._event_shape +
        #                                 self._event_shape)

    @lazy_property
    def precision_matrix(self):
        raise NotImplementedError("TODO")

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        raise NotImplementedError("TODO")

    # implemented in original ---
    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    # not implemented in original ---
    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError
