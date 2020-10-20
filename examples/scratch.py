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

from torch_hlm.mixed_effects_module import GaussianMixedEffectsModule
from torch_hlm.simulate import simulate_raneffects

torch.manual_seed(42)
np.random.seed(42)

NUM_GROUPS = 50000
NUM_RES = 10

df_train, df_raneff = simulate_raneffects(NUM_GROUPS, 50, NUM_RES + 1)
df_train['y'] += (
    1. + # intercept
    .5 * df_train['x1'] + #fixeff 
    .9 * np.random.randn(len(df_train.index)) # noise
)
df_train['time'] = df_train.groupby('group').cumcount()
df_train = df_train.merge(
    df_train.loc[:,['group']].\
        drop_duplicates().\
        assign(_max_time=lambda df: np.random.randint(low=25, high=50,size=len(df)))
).\
    query("time<=_max_time").\
    reset_index(drop=True).\
    drop(columns=['_max_time'])
df_train

X = df_train.loc[:,df_train.columns.str.startswith('x')].values.astype('float32')
y = df_train['y'].values.astype('float32')
group_ids = df_train['group'].values

# +
m = GaussianMixedEffectsModule(fixeff_features=NUM_RES, raneff_features=NUM_RES)

# _init_re_cov = .05 * torch.eye(NUM_RES+1)
# _init_re_cov[0,0] = 1.
fixed_re_cov = torch.from_numpy(df_raneff.drop(columns=['group']).cov().values.astype('float32'))
m.set_re_cov('group',fixed_re_cov)

# TODO: test `wait_for_res` with simpsons paradox

# + hideOutput=true
looper = m.fit_loop(X=X, 
                    y=y, 
                    group_ids=group_ids, 
                    optimizer=torch.optim.LBFGS([x for x in m.parameters() if not m._is_cov_param(x)], 
                                                lr=.25),#torch.optim.SGD(m.parameters(),lr=.1)
                   )

loss_history = []
patience = 0
prev_loss = float('inf')
while True:
    loss = next(looper)
    loss_history.append(loss)
    if abs(prev_loss - loss) < .01:
        patience += 1
    else:
        patience = 0
    if patience > 2:
        break
    prev_loss = loss
# -

pd.Series(loss_history).plot()

dict(m.named_parameters())

with torch.no_grad():
    print(m.re_distribution('group').covariance_matrix.numpy().round(2))

df_train['y_pred'] = m(
    X=X, 
    group_ids=group_ids,
    re_solve_data=(X, y, group_ids)
 ).numpy()

with torch.no_grad():
    df_raneff_est = pd.DataFrame(m.get_res(X,y,group_ids)['group'].numpy()).assign(group=list(range(NUM_GROUPS)))
df_raneff_est.columns = df_raneff.columns.tolist()

# +
foo = pd.concat([df_raneff.assign(type='true'),df_raneff_est.assign(type='estimate')]).\
    melt(id_vars=['group','type']).\
    pivot_table(values='value', columns='type', index=['group','variable']).\
    reset_index()

foo.groupby('variable')['estimate','true'].corr().reset_index(level=0).loc['estimate','true'].values
# -

from plotnine import *

print(
    ggplot(foo, aes(x='estimate', y='true')) + facet_wrap("~variable") + 
    geom_point(alpha=.10) + stat_smooth(color='blue') +
    geom_abline()
)

# +

print(
    ggplot(df_train.query("group.isin(group.sample(9))"), 
                          aes(x='x1')) + 
    geom_point(aes(y='y'), alpha=.10) + 
    stat_smooth(aes(y='y'), color='blue') +
    stat_smooth(aes(y='y_pred'), color='red') +
    facet_wrap("~group")
)
# -


