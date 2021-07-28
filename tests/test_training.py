import unittest

import numpy as np
import pandas as pd

from parameterized import parameterized

import torch

from torch_hlm.mixed_effects_model import MixedEffectsModel
from torch_hlm.simulate import simulate_raneffects


class TestTraining(unittest.TestCase):
    @parameterized.expand([('binary',), ('gaussian',)])
    @torch.no_grad()
    def test_training_single_gf(self,
                                response_type: str,
                                num_groups: int = 500,
                                num_res: int = 2,
                                num_obs_per_group: int = 100,
                                intercept: float = -1.,
                                noise: float = 1.1):
        print("`test_training_single_gf()` with config `{}`".format({k: v for k, v in locals().items() if k != 'self'}))
        torch.manual_seed(43)
        np.random.seed(43)

        # SIMULATE DATA -----
        df_train, df_raneff_true = simulate_raneffects(
            num_groups=num_groups,
            obs_per_group=int(num_obs_per_group * 1.25),
            num_raneffects=num_res + 1,
            # std_multi=[0.2] + [0.1] * NUM_RES
        )
        df_train['y'] += (
                intercept +
                .5 * df_train['x1'] +
                noise * np.random.randn(len(df_train.index))
        )

        if response_type == 'binary':
            df_train['y'] = (df_train['y'] > 0).astype('int')
        elif response_type == 'binomial':
            raise NotImplementedError("TODO")

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
        model = MixedEffectsModel(
            fixeff_cols=predictors,
            response_type='binomial' if response_type.startswith('bin') else 'gaussian',
            raneff_design={'group': predictors},
            response_colname='y'
        )

        if response_type.startswith('bin'):
            true_cov = torch.as_tensor(df_raneff_true.drop(columns='group').cov().values, dtype=torch.float32)
            model.fit(df_train, re_cov=true_cov)
        else:
            model.fit(df_train)

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

        self.assertTrue(
            (df_compare.groupby('variable')[['true', 'est']].corr().reset_index(0).loc['true', 'est'] > .80).all()
        )

        # TODO: fixed-effects
