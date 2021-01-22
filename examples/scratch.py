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

torch.manual_seed(43)
np.random.seed(43)

# ## Simulate Data

BINARY_RESPONSE = True
NUM_GROUPS = 1000
NUM_RES = 2
NUM_OBS_PER_GROUP = 50
OPTIMIZE_COV = True

# +
df_train, df_raneff_true = simulate_raneffects(
    num_groups=NUM_GROUPS, 
    obs_per_group=int(NUM_OBS_PER_GROUP*1.25), 
    num_raneffects=NUM_RES + 1, 
    #std_multi=[0.2] + [0.1] * NUM_RES
)
df_train['y'] += (
    -1. + # intercept
    .5 * df_train['x1'] + #fixeff 
    1.1 * np.random.randn(len(df_train.index)) # noise
)

if BINARY_RESPONSE:
    df_train['y'] = (df_train['y'] > 0).astype('int')

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
# -

# this is pretty sensitive to the random seed, due to how `simulate_raneffects` generates a random re-corr mat
df_train.groupby('group')['y'].mean().hist()

# ## Fit Model

# +
predictors = df_train.columns[df_train.columns.str.startswith('x')].tolist()

model = MixedEffectsModel(
    fixeff_cols=predictors, 
    response_type='binary' if BINARY_RESPONSE else 'gaussian',
    raneff_design={'group' : predictors}, 
    response_colname='y'
)
# -

model.module_ = model._initialize_module()

# +
# foo.state_dict()['fixed_effects_nn.bias'][:] = -1
# foo.state_dict()['fixed_effects_nn.weight'].view(-1)[0] = .5
# foo.state_dict()['fixed_effects_nn.weight'].view(-1)[1] = 0
# foo.state_dict()
# -

model.module_.set_re_cov('group', 
                         torch.as_tensor(df_raneff_true.drop(columns='group').cov().values, dtype=torch.float32))
model.module_.re_distribution().covariance_matrix

# +
# foo.get_res(*model.build_model_mats(df_train.query("group==0")))

# +
# df_raneff_true.query("group==0")

# +
# foo.get_loss(*model.build_model_mats(df_train), foo.get_res(*model.build_model_mats(df_train)))
# -

import datetime
print(datetime.datetime.now())
model.fit(
    X=df_train, 
    optimize_re_cov=False,#OPTIMIZE_COV, 
    #callbacks=[lambda x: print("Residual Var", model.module_.residual_var)],
    reset=False,
    loss_type='one_step_ahead'
)
print(datetime.datetime.now())
# -

pd.Series(model.loss_history_[-1]).plot()

model.module_.state_dict()

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
