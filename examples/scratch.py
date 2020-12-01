# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import torch

from torch_hlm.mixed_effects_model import MixedEffectsModel
from torch_hlm.simulate import simulate_raneffects

torch.manual_seed(42)
np.random.seed(42)

# ## Simulate Data

NUM_GROUPS = 150_000
NUM_RES = 10
NUM_OBS_PER_GROUP = 10
OPTIMIZE_COV = True

df_train, df_raneff_true = simulate_raneffects(NUM_GROUPS, int(NUM_OBS_PER_GROUP*1.25), NUM_RES + 1)
df_train['y'] += (
    1. + # intercept
    .5 * df_train['x1'] + #fixeff 
    .9 * np.random.randn(len(df_train.index)) # noise
)
df_train['time'] = df_train.groupby('group').cumcount()
df_train = df_train.merge(
    df_train.loc[:,['group']].\
        drop_duplicates().\
        assign(
            _max_time=lambda df: 
                np.random.randint(low=int(NUM_OBS_PER_GROUP*.75), high=int(NUM_OBS_PER_GROUP*1.25), size=len(df))
    )
).\
    query("time<=_max_time").\
    reset_index(drop=True).\
    drop(columns=['_max_time'])
df_train

# ## Fit Model

# +
predictors = df_train.columns[df_train.columns.str.startswith('x')].tolist()

model = MixedEffectsModel(
    fixeff_cols=predictors, 
    raneff_design={'group' : predictors}, 
    response_colname='y'
)
# -

import datetime
print(datetime.datetime.now())
model.fit(
    X=df_train, 
    optimize_re_cov=OPTIMIZE_COV, 
    callbacks=[lambda x: print("Residual Var", model.module_.residual_var.item())]
)
print(datetime.datetime.now())
# -

pd.Series(model.loss_history_[-1]).plot()

# ## Compare Results to Ground Truth

# ### RE Covariance

with torch.no_grad():
    _ = model.module_.re_distribution('group').covariance_matrix 
    print(_.numpy().round(2))
true_cov = torch.from_numpy(df_raneff_true.drop(columns=['group']).cov().values.astype('float32'))
print(true_cov.numpy().round(2))

# ### RE Estimates

# +
with torch.no_grad():
    df_raneff_est = pd.DataFrame(model.module_.get_res(*model.build_model_mats(df_train))['group'].numpy()).assign(group=list(range(NUM_GROUPS)))
df_raneff_est.columns = df_raneff_true.columns.tolist()

df_raneff = pd.concat([df_raneff_true.assign(type='true'),df_raneff_est.assign(type='estimate')]).\
    melt(id_vars=['group','type']).\
    pivot_table(values='value', columns='type', index=['group','variable']).\
    reset_index()

df_raneff.groupby('variable')[['estimate','true']].corr().reset_index(level=0).loc['estimate','true'].values
# -

from plotnine import *

print(
    ggplot(df_raneff.sample(n=min(5_000,NUM_GROUPS)), 
           aes(x='estimate', y='true')) + facet_wrap("~variable") + 
    geom_point(alpha=.10) + stat_smooth(color='blue') +
    geom_abline()
)

# ### Predicted vs. Actual

df_train['y_pred'] = model.predict(df_train, group_data=df_train)
print(
    ggplot(df_train.query("group.isin(group.sample(9))"), 
                          aes(x='x1')) + 
    geom_point(aes(y='y'), alpha=.10) + 
    stat_smooth(aes(y='y'), color='blue') +
    stat_smooth(aes(y='y_pred'), color='red') +
    facet_wrap("~group")
)


