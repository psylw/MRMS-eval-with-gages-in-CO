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
from model_input import model_input
df = pd.read_feather('output/train_test2')

cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)


import sys
sys.path.append('output')
from gb_q_hyp import param,idx

# %%
# plot correlation matrix of highest performing permutation

pltcorr = df.drop(columns=['start','storm_id','norm_diff']).iloc[:,list(all_permutations[idx])]

plt.figure(figsize=(8, 6))
# Generating the correlation matrix
correlation_matrix = pltcorr.corr(method='spearman')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# %%
# define untuned and tuned model

all_models = {}
for alpha, p in zip([0.05, 0.5, 0.95],param):
    gbr_t = GradientBoostingRegressor(**p,loss="quantile", alpha=alpha)
    all_models["qgb_t %1.2f" % alpha] = gbr_t
    #br = GradientBoostingRegressor(loss="quantile", alpha=alpha)
    #all_models["qgb %1.2f" % alpha] = gbr'''

#%%
scoring={'neg_05p': make_scorer(
    mean_pinball_loss,
    alpha=.05,
),
'neg_5p': make_scorer(
    mean_pinball_loss,
    alpha=.5,
),
'neg_95p': make_scorer(
    mean_pinball_loss,
    alpha=.95,
)}
# %%
scores=[]
for name, gbr in zip(all_models.keys(),list(all_models.values())):
    print(name)

    x = cross_validate(gbr,X_train[:,all_permutations[idx]],y_train, cv = cv,
                    scoring=scoring)
    print(x)
    scores.append([name,
                x['test_neg_05p'].mean(),x['test_neg_05p'].std(),
                x['test_neg_5p'].mean(),x['test_neg_5p'].std(),
                x['test_neg_95p'].mean(),x['test_neg_95p'].std(),
                ])
    
scores = pd.DataFrame(scores,columns=['names',
                              'test_neg_05p_mean','test_neg_05p_std',
                              'test_neg_5p_mean','test_neg_5p_std',
                              'test_neg_95p_mean','test_neg_95p_std'])
# %%
test_results={}
for name, gbr,a in zip(all_models.keys(),list(all_models.values()),[0.05,.5,.95]):
    print(name)
    gbr.fit(X_train[:,all_permutations[idx]],y_train)
    pred = gbr.predict(X_test[:,all_permutations[idx]])
    test_results[name]=pred
    print(round(mean_pinball_loss(y_test,pred,alpha=a),3))

test_results = pd.DataFrame(test_results)
test_results['truth'] = y_test
test_results.to_feather('output/test_results')
#test_results=test_results.divide(test.max_mrms.values,axis=0)
#%%
fig = plt.figure(figsize=(10, 8))
test_results = test_results.sort_values(by='qgb_t 0.50').reset_index(drop=True)
#plt.plot(test_results.reduced_data, test_results['qgb 0.50'], 'r+',label='median')
outliers = test_results.loc[(test_results.truth>test_results['qgb_t 0.95'])|(test_results.truth<test_results['qgb_t 0.05'])]

notoutlier = test_results.loc[(test_results.truth<test_results['qgb_t 0.95'])&(test_results.truth>test_results['qgb_t 0.05'])]

plt.plot(outliers['qgb_t 0.50'], outliers['truth'], 'r+',label='out of range')

plt.plot(notoutlier['qgb_t 0.50'], notoutlier['truth'], 'g+',label='in range')
plt.fill_between(
test_results['qgb_t 0.50'].ravel(), test_results['qgb_t 0.05'], test_results['qgb_t 0.95'], alpha=0.4, label="Predicted 90% CI"
)

#sns.kdeplot(x=test_results['qgb_t 0.50'], y=test_results['truth'], levels=5, color="red", linewidths=1)
plt.xlabel('predicted median RMSE')
plt.ylabel('RMSE')
plt.xlim(0,test_results['qgb_t 0.50'].max())
plt.ylim(0,test_results['truth'].max()+5)
plt.plot([0,100],[0,100],'k--')
plt.legend()

plt.show()

fig.savefig("output_figures/model_perf.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')
# %%

print(round(len(test_results.loc[(test_results.truth>test_results['qgb_t 0.95'])|(test_results.truth<test_results['qgb_t 0.05'])])/len(test_results),3))



# %%
train_results=[]
for i in range(5):
    d={}
    for name, gbr,a in zip(all_models.keys(),list(all_models.values()),[0.05,.5,.95]):
        c = cv[i]
        print(name)
        gbr.fit(X_train[:,all_permutations[idx]][cv[i][0]],y_train[cv[i][0]])

        pred = gbr.predict(X_train[:,all_permutations[idx]][cv[i][1]])

        d[name] = pred
        
    d = pd.DataFrame(d)
    d['truth'] = y_train[cv[i][1]] 
    d['cv'] = i
    train_results.append(d)

# %%

#%%
cv_ci=[]

for i in range(5):
    c = train_results[i]

    cv_ci.append(len(c.loc[(c.truth>c['qgb_t 0.95'])|(c.truth<c['qgb_t 0.05'])])/len(c))
# %%
print(round(np.mean(cv_ci),3))
print(round(np.std(cv_ci),3))