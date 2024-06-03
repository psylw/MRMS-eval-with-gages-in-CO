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

def hist_rmse(test_results,state_results,train,test,state, plot_name):
    state = state.copy()
    test = test.copy()
    if 'nr_rmse' in plot_name:
        a = pd.concat([train,test])
        test['nrmse'] = test.norm_diff
        a['nrmse'] = a.norm_diff
    elif 'mean_error' in plot_name:
        a = pd.concat([train,test])
        test['nrmse'] = test.norm_diff
        a['nrmse'] = a.norm_diff
    else:
        test_results=test_results.divide(test.max_mrms.values,axis=0)
        state_results=state_results.divide(state.max_mrms.values,axis=0)

        test['nrmse'] = (test.norm_diff/test.max_mrms).values
        a = pd.concat([train,test])
        a['nrmse'] = (a.norm_diff/a.max_mrms).values

    test['qgb_t 0.50'] = test_results['qgb_t 0.50'].values
    state['qgb_t 0.50'] = state_results['qgb_t 0.50'].values

    coord_state_pred = state.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']

    coord_state_pred_lat_low = coord_state_pred.loc[coord_state_pred<coord_state_pred.quantile(.1)].reset_index().mrms_lat.values
    coord_state_pred_lon_low = coord_state_pred.loc[coord_state_pred<coord_state_pred.quantile(.1)].reset_index().mrms_lon.values

    coord_state_pred_lat_high = coord_state_pred.loc[coord_state_pred>coord_state_pred.quantile(.9)].reset_index().mrms_lat.values
    coord_state_pred_lon_high = coord_state_pred.loc[coord_state_pred>coord_state_pred.quantile(.9)].reset_index().mrms_lon.values

    # ERROR AT COORDINATES WITH LOW/HIGH MEDIAN NRMSE
    fig, axes = plt.subplots(1, 2, figsize=(16*.6,8*.6), sharey=True)

    sns.kdeplot(data=test['qgb_t 0.50'],label=r'$\alpha$ = 0.50'+' predictions for test ds',color='red',ax = axes[0])

    sns.kdeplot(data=state['qgb_t 0.50'],label=r'$\alpha$ = 0.50'+' predictions for state ds',color='darkblue',ax = axes[0])

    sns.kdeplot(data=test['nrmse'],label='test ds target error',color='black', linestyle='--',ax = axes[0])

    sns.kdeplot(data=a['nrmse'],label='all target error',color='grey', linestyle='--',ax = axes[0])
    axes[0].grid(True)
    axes[0].set_xlabel('normalized RMSE',fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].set_ylabel('Density',fontsize=12)
    axes[0].set_xlim(0,1.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].text(.05, 3, '(a)', fontsize=20)
    sns.kdeplot(data=state.loc[(state.mrms_lat.isin(coord_state_pred_lat_low))&(state.mrms_lon.isin(coord_state_pred_lon_low))]['qgb_t 0.50'],label='low state '+r'$\alpha$ = 0.50'+' predictions',color='darkblue',ax=axes[1],linestyle='dotted')

    sns.kdeplot(data=state.loc[(state.mrms_lat.isin(coord_state_pred_lat_high))&(state.mrms_lon.isin(coord_state_pred_lon_high))]['qgb_t 0.50'],label='high state '+r'$\alpha$ = 0.50'+' predictions',color='darkblue', linestyle='--',ax=axes[1])

    sns.kdeplot(data=state['qgb_t 0.50'],label='all state '+r'$\alpha$ = 0.50'+' predictions',color='darkblue',ax = axes[1])

    axes[1].grid(True)

    axes[1].set_xlabel('normalized RMSE',fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].set_xlim(0,1.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].text(.05, 3, '(b)', fontsize=20)
    plt.tight_layout()

    #fig.savefig("../output_figures/experiments/figures/f02"+plot_name+".pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
    fig.suptitle(plot_name, fontsize=16)
    fig.savefig("../output/experiments/figures/f02"+plot_name+".png",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')