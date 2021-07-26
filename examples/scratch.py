# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

# %%
import torch

from torch_hlm.mixed_effects_model import MixedEffectsModel
from torch_hlm.simulate import simulate_raneffects

# %%
torch.manual_seed(43)
np.random.seed(43)

BINARY_RESPONSE = True
NUM_GROUPS = 500
NUM_RES = 2
NUM_OBS_PER_GROUP = 100
INTERCEPT = -1.
NOISE = 1.1

# +
df_train, df_raneff_true = simulate_raneffects(
    num_groups=NUM_GROUPS,
    obs_per_group=int(NUM_OBS_PER_GROUP * 1.25),
    num_raneffects=NUM_RES + 1,
    # std_multi=[0.2] + [0.1] * NUM_RES
)
df_train['y'] += (
        INTERCEPT +  # intercept
        .5 * df_train['x1'] +  # fixeff
        NOISE * np.random.randn(len(df_train.index))  # noise
)

if BINARY_RESPONSE:
    df_train['y'] = (df_train['y'] > 0).astype('int')

df_train['time'] = df_train.groupby('group').cumcount()
df_train = df_train.merge(
    df_train.loc[:, ['group']]. \
        drop_duplicates(). \
        assign(
        _max_time=lambda df:
        np.random.randint(low=int(NUM_OBS_PER_GROUP * .75), high=int(NUM_OBS_PER_GROUP * 1.25), size=len(df))
    )
). \
    query("time<=_max_time"). \
    reset_index(drop=True). \
    drop(columns=['_max_time'])
df_train

# %%
true_cov = torch.as_tensor(df_raneff_true.drop(columns='group').cov().values, dtype=torch.float32)
true_cov

# %%
predictors = df_train.columns[df_train.columns.str.startswith('x')].tolist()

model = MixedEffectsModel(
    fixeff_cols=predictors,
    response_type='binomial' if BINARY_RESPONSE else 'gaussian',
    raneff_design={'group' : predictors},
    response_colname='y',
    #optimizer_kwargs={'lr':.0001}
)

# %%
# model.module_ = model._initialize_module()
# model.module_.state_dict()

# %%
model.fit(df_train,
          #callbacks=[lambda x: print(model.module_.fixed_effects_nn.state_dict())],
          re_cov=true_cov if BINARY_RESPONSE else True)

# %%
with torch.no_grad():
    df_raneff_est = pd.DataFrame(model.module_.get_res(*model.build_model_mats(df_train))['group'].numpy(),
                                columns=df_raneff_true.columns[0:-1])
    df_raneff_est['group'] = df_raneff_true['group']
df_raneff_est

# %%


# %%
df_compare = pd.concat([df_raneff_est.assign(type='est'), df_raneff_true.assign(type='true')]).\
    melt(id_vars=['group','type']).\
    pivot(index=['group','variable'], columns='type', values='value').\
    reset_index()
df_compare.columns.name = None

# from plotnine import *
# print(ggplot(df_compare, aes(x='true', y='est')) +
#       facet_wrap("~variable", scales='free') +
#       geom_point(alpha=.05) +
#       stat_smooth() +
#       geom_abline())

# %%
df_compare.groupby('variable')[['true','est']].corr().reset_index(0).loc['true','est']

# %%
