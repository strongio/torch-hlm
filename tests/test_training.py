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


class TestTraining(unittest.TestCase):

    @parameterized.expand([
        ('gaussian', [0, 0]),
        ('binary', [0, 0])
    ])
    def test_training_multiple_gf(self,
                                  response_type: str,
                                  num_res: Sequence[int],
                                  intercept: float = -1.,
                                  noise: float = 1.0):
        print("\n`test_training_multiple_gf()` with config `{}`".
              format({k: v for k, v in locals().items() if k != 'self'}))
        torch.manual_seed(0)
        np.random.seed(43)

        # SIMULATE DATA -----
        df_train = []
        df_raneff_true = []
        for i, num_res_g in enumerate(num_res):
            df_train_g, df_raneff_true_g = simulate_raneffects(
                num_groups=40,
                obs_per_group=1,
                num_raneffects=num_res_g + 1,
                std_multi=.25
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

        if response_type == 'binary':
            df_train['y'] = np.random.binomial(p=expit(df_train['y'].values), n=1)
        elif response_type == 'binomial':
            raise NotImplementedError("TODO")
        else:
            df_train['y'] += noise * np.random.randn(df_train.shape[0])

        # FIT MODEL -----
        covariance = {}
        for gf, df_raneff_true_g in df_raneff_true.groupby('gf'):
            covariance[gf] = torch.as_tensor(df_raneff_true_g.drop(columns=['group', 'gf']).cov().values)

        raneff_design = {f"g{i + 1}": [] for i in range(len(num_res))}
        for gf in list(raneff_design):
            raneff_design[gf] = df_train.columns[df_train.columns.str.startswith(gf + '_x')].tolist()
        model = MixedEffectsModel(
            fixeff_cols=[],
            response_type='binomial' if response_type.startswith('bin') else 'gaussian',
            raneff_design=raneff_design,
            response='y',
            covariance=covariance,
            loss_type='iid'
        )
        model.fit(df_train)

        # COMPARE TRUE vs. EST -----
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
            self.assertGreater(df_corr.loc['true', 'est'].min(), .6 if response_type.startswith('bin') else .8)
            self.assertGreater(df_corr.loc['true', 'est'].mean(), .7 if response_type.startswith('bin') else .9)
        except AssertionError:
            print(df_corr)
            raise

        self.assertLess(abs(model.module_.fixed_effects_nn.bias - intercept), .13 * len(num_res))
        wt = model.module_.fixed_effects_nn.weight.squeeze()
        if len(wt):
            self.assertLess(abs(wt[0] - .5), .1)
            self.assertLess(wt[1:].abs().max(), .1)

    @parameterized.expand([
        # ('binary', 'cv'),
        ('binary', None),
        # ('gaussian', 'cv'),
        ('gaussian', None)
    ])
    def test_training_single_gf(self,
                                response_type: str,
                                loss_type: Optional[str] = None,
                                num_groups: int = 500,
                                num_res: int = 2,
                                num_obs_per_group: int = 100,
                                intercept: float = -1.,
                                noise: float = 1.0):
        print("\n`test_training_single_gf()` with config `{}`".
              format({k: v for k, v in locals().items() if k != 'self'}))
        torch.manual_seed(43)
        np.random.seed(43)

        # SIMULATE DATA -----
        df_train, df_raneff_true = simulate_raneffects(
            num_groups=num_groups,
            obs_per_group=int(num_obs_per_group * 1.25),
            num_raneffects=num_res + 1,
            # std_multi=[0.2] + [0.1] * NUM_RES
        )
        assert num_res > 0  # we assume x1 exists in this sim
        df_train['y'] += (intercept + .5 * df_train['x1'])

        if response_type == 'binary':
            df_train['y'] = np.random.binomial(p=expit(df_train['y'].values), n=1)
        elif response_type == 'binomial':
            raise NotImplementedError("TODO")
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
        if response_type.startswith('bin') and loss_type != 'cv':
            print("will *not* optimize covariance")
            covariance = torch.as_tensor(df_raneff_true.drop(columns='group').cov().values)
        else:
            print("*will* optimize covariance")
        model = MixedEffectsModel(
            fixeff_cols=predictors,
            response_type='binomial' if response_type.startswith('bin') else 'gaussian',
            raneff_design={'group': predictors},
            response='y',
            covariance=covariance,
            loss_type=loss_type
        )
        model.fit(df_train)  # , callbacks=[lambda x: print(model.module_.state_dict())])

        # COMPARE TRUE vs. EST -----
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
            self.assertGreater(df_corr.loc['true', 'est'].min(), .5 if response_type.startswith('bin') else .6)
            self.assertGreater(df_corr.loc['true', 'est'].mean(), .6 if response_type.startswith('bin') else .8)
        except AssertionError:
            print(df_corr)
            raise

        self.assertLess(abs(model.module_.fixed_effects_nn.bias - intercept), .1)
        wt = model.module_.fixed_effects_nn.weight.squeeze()
        self.assertLess(abs(wt[0] - .5), .1)
        self.assertLess(wt[1:].abs().max(), .1)
