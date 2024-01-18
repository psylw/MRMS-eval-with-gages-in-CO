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
import seaborn as sns
fig, ax = plt.subplots()
sns.kdeplot(data=test_results['qgb_t 0.50'],label='test dataset predicted 0.5 q')
all = pd.concat([train,test])
sns.kdeplot(data=all.norm_diff/all.max_mrms,label='all target values')
sns.kdeplot(data=state_results['qgb_t 0.50'],label='statewide dataset predicted 0.5 q')

plt.xlabel('nRMSE')
plt.legend()
plt.xlim(0,2)
#%%
fig.savefig("../output_figures/hist_nRMSE.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')
# %%
