import functools
import unittest
from typing import Sequence, Optional

import numpy as np
import pandas as pd

from parameterized import parameterized

import torch
from scipy.special import expit

from torch_hlm.mixed_effects_model import MixedEffectsModel
from torch_hlm.simulate import simulate_raneffects

SEED = 2021 - 8 - 1


class TestTraining(unittest.TestCase):

    @parameterized.expand([
        ('gaussian', [0, 0, 0]),
        ('binomial', [0, 0])
    ], skip_on_empty=True)
    def test_training_multiple_gf(self,
                                  response_type: str,
                                  num_res: Sequence[int],
                                  intercept: float = -1.,
                                  noise: float = 1.0):
        print("\n`test_training_multiple_gf()` with config `{}`".
              format({k: v for k, v in locals().items() if k != 'self'}))
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # SIMULATE DATA -----
        df_train = []
        df_raneff_true = []
        for i, num_res_g in enumerate(num_res):
            df_train_g, df_raneff_true_g = simulate_raneffects(
                num_groups=40,
                obs_per_group=5 if len(num_res) <= 2 else 1,
                num_raneffects=num_res_g + 1,
            )
            df_train.append(df_train_g.rename(columns={'y': f"g{i + 1}_y", 'group': f'g{i + 1}'}))
            df_raneff_true.append(df_raneff_true_g.assign(gf=f"g{i + 1}"))

        #
        df_train = functools.reduce(lambda x, y: x.merge(y, how='cross'), df_train)
        df_train['y'] = intercept
        for i in range(len(num_res)):
            df_train['y'] += df_train.pop(f"g{i + 1}_y")

        #
        df_raneff_true = pd.concat(df_raneff_true).reset_index(drop=True)

        df_train['n'] = np.random.poisson(5, size=df_train.shape[0]) + 1
        if response_type.startswith('bin'):
            if response_type == 'binary':
                df_train['n'] = 1
            df_train['y'] = np.random.binomial(p=expit(df_train['y'].values), n=df_train['n']) / df_train['n']
        else:
            df_train['y'] += noise * np.random.randn(df_train.shape[0]) / df_train['n']

        # FIT MODEL -----
        # covariance = {torch.as_tensor(df_raneff_true_g.drop(columns=['group', 'gf']).dropna(axis=1).cov().values)
        #               for gf, df_raneff_true_g in df_raneff_true.groupby('gf')}
        raneff_design = {f"g{i + 1}": [f'x{_ + 1}' for _ in range(n)] for i, n in enumerate(num_res)}
        model = MixedEffectsModel(
            fixeff_cols=[],
            response_type='binomial' if response_type.startswith('bin') else 'gaussian',
            raneff_design=raneff_design,
            response_col='y',
            weight_col='n',
            loss_type='mc',
        )
        model.fit(df_train)

        # COMPARE TRUE vs. EST -----
        self.assertLess(abs(model.module_.fixed_effects_nn.bias - intercept), .13 * len(num_res))
        wt = model.module_.fixed_effects_nn.weight.squeeze()
        assert not len(wt)  # haven't set up unit-tests for this

        with torch.no_grad():
            res_per_g = model.module_.get_res(*model.build_model_mats(df_train))
            df_raneff_est = []
            for gf, re_mat in res_per_g.items():
                df_raneff_est.append(
                    pd.DataFrame(re_mat.numpy(), columns=[f'x{i}' for i in range(num_res[i] + 1)]).assign(gf=gf)
                )
                df_raneff_est[-1]['group'] = np.unique(df_train[gf])
            df_raneff_est = pd.concat(df_raneff_est).reset_index(drop=True)
        df_compare = pd.concat([df_raneff_est.assign(type='est'), df_raneff_true.assign(type='true')]). \
            melt(id_vars=['group', 'gf', 'type']). \
            pivot(index=['group', 'gf', 'variable'], columns='type', values='value'). \
            reset_index()
        df_compare.columns.name = None
        df_corr = df_compare.groupby(['variable', 'gf'])[['true', 'est']].corr().reset_index([0, 1])
        try:
            # these are very sensitive to exact config (e.g. num_groups), may want to laxen
            self.assertGreater(df_corr.loc['true', 'est'].min(), .5 if response_type.startswith('bin') else .7)
            self.assertGreater(df_corr.loc['true', 'est'].mean(), .6 if response_type.startswith('bin') else .8)
        except AssertionError:
            print(df_corr)
            raise

    @parameterized.expand([
        ('binary', 'cv'),
        ('binary', 'iid'),
        ('binomial', 'iid'),
        ('gaussian', 'cv'),
        ('gaussian', 'mvnorm'),
        ('gaussian', 'iid')
    ])
    def test_training_single_gf(self,
                                response_type: str,
                                loss_type: Optional[str] = None,
                                num_res: int = 2,
                                intercept: float = -1,
                                num_groups: int = 500,
                                noise: float = 1.0):
        print("\n`test_training_single_gf()` with config `{}`".
              format({k: v for k, v in locals().items() if k != 'self'}))
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # SIMULATE DATA -----
        df_train, df_raneff_true = simulate_raneffects(
            num_groups=num_groups,
            obs_per_group=int(num_obs_per_group * 1.25),
            num_raneffects=num_res + 1
        )
        assert num_res > 0  # we assume x1 exists in this sim
        df_train['y'] += (intercept + .5 * df_train['x1'])

        df_train['n'] = np.random.poisson(5, size=df_train.shape[0]) + 1
        if response_type.startswith('bin'):
            if response_type == 'binary':
                df_train['n'] = 1
            df_train['y'] = np.random.binomial(p=expit(df_train['y'].values), n=df_train['n']) / df_train['n']
        else:
            df_train['y'] += noise * np.random.randn(df_train.shape[0])

        df_train['time'] = df_train.groupby('group').cumcount()
        df_train = df_train.merge(
            df_train.loc[:, ['group']]. \
                drop_duplicates(). \
                assign(
                _max_time=lambda df:
                np.random.randint(low=int(num_obs_per_group * .75), high=int(num_obs_per_group * 1.25), size=len(df))
            )
        ). \
            query("time<=_max_time"). \
            reset_index(drop=True). \
            drop(columns=['_max_time'])

        # FIT MODEL -----
        predictors = df_train.columns[df_train.columns.str.startswith('x')].tolist()
        covariance = 'log_cholesky'
        if loss_type in ('mvnorm', 'cv'):
            print("*will* optimize covariance")
        else:
            print("will *not* optimize covariance")
            covariance = torch.as_tensor(df_raneff_true.drop(columns='group').cov().values)
        model = MixedEffectsModel(
            fixeff_cols=predictors,
            response_type='binomial' if response_type.startswith('bin') else 'gaussian',
            raneff_design={'group': predictors},
            response_col='y',
            weight_col='n',
            covariance=covariance,
            loss_type=loss_type
        )
        model.fit(df_train)

        # COMPARE TRUE vs. EST -----
        self.assertLess(abs(model.module_.fixed_effects_nn.bias - intercept), .1)
        wt = model.module_.fixed_effects_nn.weight.squeeze()
        self.assertLess(abs(wt[0] - .5), .1)
        self.assertLess(wt[1:].abs().max(), .1)

        with torch.no_grad():
            df_raneff_est = pd.DataFrame(model.module_.get_res(*model.build_model_mats(df_train))['group'].numpy(),
                                         columns=df_raneff_true.columns[0:-1])
            df_raneff_est['group'] = df_raneff_true['group']

        df_compare = pd.concat([df_raneff_est.assign(type='est'), df_raneff_true.assign(type='true')]). \
            melt(id_vars=['group', 'type']). \
            pivot(index=['group', 'variable'], columns='type', values='value'). \
            reset_index()
        df_compare.columns.name = None
        df_corr = df_compare.groupby('variable')[['true', 'est']].corr().reset_index(0)
        try:
            # these are very sensitive to exact config (e.g. num_groups), may want to laxen
            self.assertGreater(df_corr.loc['true', 'est'].min(), .35 if response_type.startswith('bin') else .45)
            self.assertGreater(df_corr.loc['true', 'est'].mean(), .5 if response_type.startswith('bin') else .7)
        except AssertionError:
            print(df_corr)
            raise
