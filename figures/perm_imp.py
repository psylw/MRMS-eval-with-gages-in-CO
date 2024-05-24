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
sys.path.append('../utils')
from model_input import model_input

df = pd.read_feather('../output/train_test2')
############################################################
# experiments
# add year
#df['year'] = [df.start[i].year for i in df.index]
# train post v12
#df = df[df.year>=2021]
# other metrics
df = df.loc[df.total_mrms_accum>1].reset_index(drop=True)
df = df.dropna()
df['norm_diff'] = pd.read_feather('../output/mean_error')
#%%
###########################################################
sys.path.append('../output')
sys.path.append('../test')
cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)

from gb_q_hyp_mean_bias import param, idx
# %%
# define untuned and tuned model
all_models = {}
for alpha, p in zip([0.05, 0.5, 0.95],param[0:3]):
    gbr_t = GradientBoostingRegressor(**p,loss="quantile", alpha=alpha)
    all_models["qgb_t %1.2f" % alpha] = gbr_t


# %%
corr = ['total_mrms_accum',
'max_mrms',
'max_accum_atgage',
#'median_int_point',
'std_int_point',
'var_int_point',
'mean_int_point' ,
'std_accum_point',
'var_accum_point',
'mean_accum_point',
'median_accum_point']

from sklearn.inspection import permutation_importance
imp = []
for name, gbr,a in zip(all_models.keys(),list(all_models.values()),[0.05,.5,.95]):
    gbr.fit(X_train[:,all_permutations[idx]], y_train)


    result_train = permutation_importance(
        gbr, X_train[:,all_permutations[idx]], y_train, n_repeats=10,
         random_state=42, n_jobs=2,scoring=make_scorer(mean_pinball_loss,
    alpha=a,greater_is_better=False))
    result_test = permutation_importance(
        #gbr, X_train[:,all_permutations[i]], y_train, n_repeats=10,
        gbr, X_test[:,all_permutations[idx]], y_test, n_repeats=10, random_state=42, n_jobs=2,scoring=make_scorer(mean_pinball_loss,
    alpha=a,greater_is_better=False))

    sorted_importances_idx_train = result_train.importances_mean.argsort()
    sorted_importances_idx_test = result_test.importances_mean.argsort()

    importances_train = pd.DataFrame(
        result_train.importances[sorted_importances_idx_train].T,
        columns=train.drop(columns='norm_diff').iloc[:,list(all_permutations[idx])].columns[sorted_importances_idx_train]
    )
    importances_test = pd.DataFrame(
        result_test.importances[sorted_importances_idx_test].T,
        columns=train.drop(columns='norm_diff').iloc[:,list(all_permutations[idx])].columns[sorted_importances_idx_test]
    )

    importances_train = importances_train.drop(columns=corr,errors='ignore')
    importances_train = importances_train.rename(columns={'median_int_point':'median int','rqi_std':'RQI std dev', 'rqi_min': 'RQI min',
       'point_aspect':'aspect', 'mrms_lat':'latitude', 'point_elev':'elevation'})
    importances_test = importances_test.drop(columns=corr,errors='ignore')
    importances_test = importances_test.rename(columns={'median_int_point':'median int','rqi_std':'RQI std dev', 'rqi_min': 'RQI min',
       'point_aspect':'aspect', 'mrms_lat':'latitude', 'point_elev':'elevation'})
    importances_train = importances_train.T.stack().reset_index().drop(columns='level_1').rename(columns={'level_0':'feature',0:'pi'})
    importances_train['model'] = name
    importances_train['set'] = 'train'

    importances_test = importances_test.T.stack().reset_index().drop(columns='level_1').rename(columns={'level_0':'feature',0:'pi'})
    importances_test['model'] = name
    importances_test['set'] = 'test'

    importances = pd.concat([importances_test,importances_train])
    imp.append(importances)

#%%
# Create a 1 by 3 subplot
fig, axes = plt.subplots(1, 3, figsize=(20*.65, 3*.65), sharey=True) # Adjust the figure size as needed

# Subplot 1

sns.barplot(data=imp[0], x='feature', y='pi', hue='set', palette='Set1',errorbar="sd",ax=axes[0])
axes[0].legend([], [], frameon=False)  # Remove legend
axes[0].set_ylabel('')  # Remove x-axis label
axes[0].set_xlabel('')  # Remove x-axis label
axes[0].set_title(r'$\alpha$ = 0.05',fontsize=12)
lbl = [item.get_text() for item in axes[0].get_xticklabels()]
axes[0].set_xticklabels(lbl, rotation=60,fontsize=12)
axes[0].grid(True)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
desired_y_ticks = [0, .025, .05, .075,.1]  # Replace with your desired values
axes[0].set_yticks(desired_y_ticks,fontsize=12)

# Subplot 2

sns.barplot(data=imp[1], x='feature', y='pi', hue='set', palette='Set1',errorbar="sd", ax=axes[1])
axes[1].legend([], [], frameon=False)  # Remove legend
axes[1].set_ylabel('')  # Remove x-axis label
axes[1].set_xlabel('')  # Remove x-axis label
axes[1].set_title(r'$\alpha$ = 0.50',fontsize=12)
lbl = [item.get_text() for item in axes[1].get_xticklabels()]
axes[1].set_xticklabels(lbl, rotation=60,fontsize=12)
axes[1].tick_params(left = False) 
axes[1].grid(True)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
# Subplot 3

sns.barplot(data=imp[2], x='feature', y='pi', hue='set', palette='Set1',errorbar="sd", ax=axes[2])

axes[2].legend([], [], frameon=False)  # Remove legend
axes[2].set_ylabel('')  # Remove x-axis label
axes[2].set_xlabel('')  # Remove x-axis label
axes[2].set_title(r'$\alpha$ = 0.95',fontsize=12)
lbl = [item.get_text() for item in axes[2].get_xticklabels()]
axes[2].set_xticklabels(lbl, rotation=60,fontsize=12)
axes[2].tick_params(left = False)
axes[2].grid(True) 
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['left'].set_visible(False)
# Show the plot
fig.text(0.07, 0.5, 'increase in mean (QL)', va='center', rotation='vertical',fontsize=12)
plt.subplots_adjust(wspace=0.06, hspace=0.02)
plt.show()
#%%
fig.savefig("../output_figures/f04.pdf",
       bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
