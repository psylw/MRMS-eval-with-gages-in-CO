#%%
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
sys.path.append('../utils')
sys.path.append('../output')

state = pd.read_feather('../output/experiments/stateclean_year')

state_results = pd.read_feather('../output/state_results')
from gb_q_hyp import param,idx

# get normalized state results
state_results=state_results.divide(state.max_mrms.values,axis=0)

from model_input import model_input
df = pd.read_feather('../output/train_test2')
cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)

state = state.copy()
state['qgb_t 0.50'] = state_results['qgb_t 0.50'].values
readable_names = {
'std_int_point': 'intensity std dev',
'median_int_point': 'intensity median',
'duration':'duration',
'mrms_lat': 'latitude', 
'point_elev': 'elevation',
'point_aspect': 'aspect',
'rqi_min': 'RQI min',
'rqi_std': 'RQI std dev',
}
state = state.rename(columns=readable_names)
#%%
# Calculate the quantiles
top_quantile = state['qgb_t 0.50'].quantile(0.9)
bottom_quantile = state['qgb_t 0.50'].quantile(0.1)
print(top_quantile)
print(bottom_quantile)

# Filter the DataFrame for the top and bottom 10%
state_bad = state.loc[state['qgb_t 0.50'] >= top_quantile]
state_good = state.loc[state['qgb_t 0.50'] <= bottom_quantile]
print(len(state_bad))
print(len(state_good))
#%%
state_bad = state_bad.drop(columns='qgb_t 0.50')
state_good = state_good.drop(columns='qgb_t 0.50')


#%%
fig, axs = plt.subplots(2,5, figsize=(14*.75,5*.85), facecolor='w', edgecolor='k',sharex=True)
fig.subplots_adjust(hspace = .18, wspace=.4)

axs = axs.ravel()

make_log = {
'latitude' : False, 
'longitude' : False, 
'total accum' : True, 
'RQI mean' : False, 
'RQI median' : False,
'RQI min' : False,
'RQI max' : False, 
'RQI std dev' : False, 
'max intensity' : True,
'max accum' : True,
'intensity median' : True,
'intensity std dev' : True,
'var intensity' : False,
'mean intensity' : True,
'median accum' : True,
'std dev accum' : False,
'var accum' : False,
'mean accum' : True,
'elevation' : False,
'slope' : False,
'aspect' : False,
'storm temp var' : False,
'storm spatial var' : False,
'duration' : False,
'month' : False,
'hour' : False,
'velocity' : False,
'area' : True
}

columns = train.drop(columns='norm_diff').iloc[:,list(all_permutations[idx])].rename(columns=readable_names)

columns = columns.drop(columns=['month','hour','year'], errors='ignore')
columns = columns.reindex(columns = [ 'intensity std dev','intensity median','latitude','duration','elevation','aspect','RQI min','RQI std dev','area','velocity']).columns

flierprops = dict(marker='o', markerfacecolor='r', markersize=4, linestyle='none')
for i,col in enumerate(columns):
        d = [state_good[col],state_bad[col]]

        axs[i].boxplot(d, flierprops=flierprops, patch_artist=True, 
           boxprops=dict(facecolor='lightblue', color='blue'),
           medianprops=dict(color='red', linewidth=2),
           whiskerprops=dict(color='blue', linewidth=1.5),
           capprops=dict(color='blue', linewidth=1.5))
        
        axs[i].set_xticks([1,2],labels=['low',
                                            'high'],rotation=45,fontsize=12)
        #axs[i].set_yticks(fontsize=12)
        axs[i].set_title(col,fontsize=12)

        axs[i].grid(True, linestyle='--', linewidth=0.5, axis='y')
        
        if make_log[col] == True:
                axs[i].set_yscale('log')
plt.tight_layout()
fig.savefig("../output_figures/experiments/S"+str(23)+".pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')