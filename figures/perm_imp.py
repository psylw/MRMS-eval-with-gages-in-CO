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
#%%
# look at cv
for i in range(len(all_permutations[25])):
    print(i)
    for name, gbr in zip(all_models.keys(),list(all_models.values())):
        
        p = list(all_permutations[25])
        p.pop(i)
        x = cross_validate(gbr,X_train[:,p],y_train, cv = cv,
                        scoring=scoring)
        print([x['test_neg_05p'].mean(),x['test_neg_5p'].mean(),x['test_neg_95p'].mean()])
# %%
corr = ['total_mrms_accum',
'max_mrms',
'max_accum_atgage',
'median_int_point',
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
    importances_train = importances_train.rename(columns={'rqi_std':'RQI std dev', 'rqi_min': 'RQI min',
       'point_aspect':'aspect', 'mrms_lat':'latitude', 'point_elev':'elevation'})
    importances_test = importances_test.drop(columns=corr,errors='ignore')
    importances_test = importances_test.rename(columns={'rqi_std':'RQI std dev', 'rqi_min': 'RQI min',
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
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True) # Adjust the figure size as needed

# Subplot 1

sns.barplot(data=imp[0], x='feature', y='pi', hue='set', palette='Set1',errorbar="sd",ax=axes[0])
axes[0].legend([], [], frameon=False)  # Remove legend
axes[0].set_ylabel('')  # Remove x-axis label
axes[0].set_xlabel('')  # Remove x-axis label
axes[0].set_title(r'$\alpha$ = 0.05')
lbl = [item.get_text() for item in axes[0].get_xticklabels()]
axes[0].set_xticklabels(lbl, rotation=60)
# Subplot 2

sns.barplot(data=imp[1], x='feature', y='pi', hue='set', palette='Set1',errorbar="sd", ax=axes[1])
axes[1].legend([], [], frameon=False)  # Remove legend
axes[1].set_ylabel('')  # Remove x-axis label
axes[1].set_xlabel('')  # Remove x-axis label
axes[1].set_title(r'$\alpha$ = 0.50')
lbl = [item.get_text() for item in axes[1].get_xticklabels()]
axes[1].set_xticklabels(lbl, rotation=60)
axes[1].tick_params(left = False) 
# Subplot 3

sns.barplot(data=imp[2], x='feature', y='pi', hue='set', palette='Set1',errorbar="sd", ax=axes[2])

axes[2].legend([], [], frameon=False)  # Remove legend
axes[2].set_ylabel('')  # Remove x-axis label
axes[2].set_xlabel('')  # Remove x-axis label
axes[2].set_title(r'$\alpha$ = 0.95')
lbl = [item.get_text() for item in axes[2].get_xticklabels()]
axes[2].set_xticklabels(lbl, rotation=60)
axes[2].tick_params(left = False) 
# Show the plot
fig.text(0.07, 0.5, 'increase in mean quantile loss', va='center', rotation='vertical')
plt.subplots_adjust(wspace=0.06, hspace=0.02)
plt.show()
#%%
fig.savefig("../output_figures/fi.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')
# %%
