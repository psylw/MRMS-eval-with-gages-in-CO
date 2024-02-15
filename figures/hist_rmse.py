# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, randint
import seaborn as sns
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV

from sklearn.metrics import  mean_absolute_error,r2_score,mean_pinball_loss, mean_squared_error,mean_pinball_loss
sys.path.append('..')
from model_input import model_input
df = pd.read_feather('../output/train_test2')
sys.path.append('../output')
sys.path.append('../test')
cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)

from gb_q_hyp import param, idx
# %%
# define untuned and tuned model
all_models = {}
for alpha, p in zip([0.05, 0.5, 0.95],param[0:3]):
    gbr_t = GradientBoostingRegressor(**p,loss="quantile", alpha=alpha)
    all_models["qgb_t %1.2f" % alpha] = gbr_t

test_results = pd.read_feather('../output/test_results')
test_results=test_results.divide(test.max_mrms.values,axis=0)
# %%
state = pd.read_feather('../output/stateclean')

scaler = StandardScaler()
X_state = scaler.fit_transform(state)

state_results={}
for name, gbr,idx in zip(all_models.keys(),list(all_models.values()),[26,45,45]):
    print(name)
    gbr.fit(X_train[:,all_permutations[idx]],y_train)
    pred = gbr.predict(X_state[:,all_permutations[idx]])
    state_results[name]=pred
state_results = pd.DataFrame(state_results)
state_results.to_feather('../output/state_results')
state_results=state_results.divide(state.max_mrms.values,axis=0)
# %%
'''import seaborn as sns
fig, ax = plt.subplots()
sns.kdeplot(data=test_results['qgb_t 0.50'],label=r'$\alpha$ = 0.50'+' predictions for test ds',color='darkgreen')

sns.kdeplot(data=state_results['qgb_t 0.50'],label=r'$\alpha$ = 0.50'+' predictions for state ds',color='darkblue')
a = pd.concat([train,test])
sns.kdeplot(data=test.norm_diff/test.max_mrms,label='test ds target values',color='black', linestyle='--')
ax.grid(True)
sns.kdeplot(data=a.norm_diff/a.max_mrms,label='all target values',color='grey', linestyle='--')
ax.grid(True)

plt.xlabel('normalized RMSE')
plt.legend()
plt.xlim(0,1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#%%
fig.savefig("../output_figures/hist_nRMSE.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')'''

# %%
test['qgb_t 0.50'] = test_results['qgb_t 0.50'].values
state['qgb_t 0.50'] = state_results['qgb_t 0.50'].values

test['nrmse'] = (test.norm_diff/test.max_mrms).values
a = pd.concat([train,test])
a['nrmse'] = (a.norm_diff/a.max_mrms).values

#%%
'''coord_test_pred = test.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']

coord_test_pred_lat_low = coord_test_pred.loc[coord_test_pred<coord_test_pred.quantile(.1)].reset_index().mrms_lat.values
coord_test_pred_lon_low = coord_test_pred.loc[coord_test_pred<coord_test_pred.quantile(.1)].reset_index().mrms_lon.values

coord_test_pred_lat_high = coord_test_pred.loc[coord_test_pred>coord_test_pred.quantile(.9)].reset_index().mrms_lat.values
coord_test_pred_lon_high = coord_test_pred.loc[coord_test_pred>coord_test_pred.quantile(.9)].reset_index().mrms_lon.values'''


coord_state_pred = state.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']

coord_state_pred_lat_low = coord_state_pred.loc[coord_state_pred<coord_state_pred.quantile(.1)].reset_index().mrms_lat.values
coord_state_pred_lon_low = coord_state_pred.loc[coord_state_pred<coord_state_pred.quantile(.1)].reset_index().mrms_lon.values

coord_state_pred_lat_high = coord_state_pred.loc[coord_state_pred>coord_state_pred.quantile(.9)].reset_index().mrms_lat.values
coord_state_pred_lon_high = coord_state_pred.loc[coord_state_pred>coord_state_pred.quantile(.9)].reset_index().mrms_lon.values
'''
coord_all_true = a.groupby(['mrms_lat','mrms_lon']).median()['nrmse']

coord_all_true_lat_low = coord_all_true.loc[coord_all_true<coord_all_true.quantile(.25)].reset_index().mrms_lat.values
coord_all_true_lon_low = coord_all_true.loc[coord_all_true<coord_all_true.quantile(.25)].reset_index().mrms_lon.values

coord_all_true_lat_high = coord_all_true.loc[coord_all_true>coord_all_true.quantile(.75)].reset_index().mrms_lat.values
coord_all_true_lon_high = coord_all_true.loc[coord_all_true>coord_all_true.quantile(.75)].reset_index().mrms_lon.values
coord_test_true = a.groupby(['mrms_lat','mrms_lon']).median()['nrmse']

coord_test_true_lat_low = coord_test_true.loc[coord_test_true<coord_test_true.quantile(.1)].reset_index().mrms_lat.values
coord_test_true_lon_low = coord_test_true.loc[coord_test_true<coord_test_true.quantile(.1)].reset_index().mrms_lon.values

coord_test_true_lat_high = coord_test_true.loc[coord_test_true>coord_test_true.quantile(.9)].reset_index().mrms_lat.values
coord_test_true_lon_high = coord_test_true.loc[coord_test_true>coord_test_true.quantile(.9)].reset_index().mrms_lon.values'''

#%%
# ERROR AT COORDINATES WITH LOW/HIGH MEDIAN NRMSE
fig, axes = plt.subplots(1, 2, figsize=(16*.6,8*.6), sharey=True)

sns.kdeplot(data=test['qgb_t 0.50'],label=r'$\alpha$ = 0.50'+' predictions for test ds',color='darkgreen',ax = axes[0])

sns.kdeplot(data=state['qgb_t 0.50'],label=r'$\alpha$ = 0.50'+' predictions for state ds',color='darkblue',ax = axes[0])

sns.kdeplot(data=test['nrmse'],label='test ds target error',color='black', linestyle='--',ax = axes[0])

sns.kdeplot(data=a['nrmse'],label='all target error',color='grey', linestyle='--',ax = axes[0])
axes[0].grid(True)
axes[0].set_xlabel('normalized RMSE')
axes[0].legend()
axes[0].set_xlim(0,1.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].text(.05, 3, '(a)', fontsize=20)
sns.kdeplot(data=state.loc[(state.mrms_lat.isin(coord_state_pred_lat_low))&(state.mrms_lon.isin(coord_state_pred_lon_low))]['qgb_t 0.50'],label='low state '+r'$\alpha$ = 0.50'+' predictions',color='darkblue',ax=axes[1],linestyle='dotted')

#sns.kdeplot(data=a.loc[(a.mrms_lat.isin(coord_all_true_lat_low))&(a.mrms_lon.isin(coord_all_true_lon_low))]['nrmse'],label='low target error',color='grey',ax=axes[1])

sns.kdeplot(data=state.loc[(state.mrms_lat.isin(coord_state_pred_lat_high))&(state.mrms_lon.isin(coord_state_pred_lon_high))]['qgb_t 0.50'],label='high state '+r'$\alpha$ = 0.50'+' predictions',color='darkblue', linestyle='--',ax=axes[1])

sns.kdeplot(data=state['qgb_t 0.50'],label='all state '+r'$\alpha$ = 0.50'+' predictions',color='darkblue',ax = axes[1])

#sns.kdeplot(data=a.loc[(a.mrms_lat.isin(coord_all_true_lat_high))&(a.mrms_lon.isin(coord_all_true_lon_high))]['nrmse'],label='high target error',color='grey', linestyle='--',ax=axes[1])

axes[1].grid(True)

axes[1].set_xlabel('normalized RMSE')
axes[1].legend()
axes[1].set_xlim(0,1.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].text(.05, 3, '(b)', fontsize=20)
plt.tight_layout()
fig.savefig("../output_figures/hist_nRMSE.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')
#%%
print(len(coord_all_true_lat_high))
print(len(coord_all_true_lat_low))
print(len(coord_all_true))
print(len(coord_state_pred_lat_high))
print(len(coord_state_pred_lat_low))
print(len(coord_state_pred))
# %%
