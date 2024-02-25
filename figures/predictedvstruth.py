#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
test_results = pd.read_feather('../output/test_results')


fig = plt.figure(figsize=(10*.6, 8*.6))
test_results = test_results.sort_values(by='qgb_t 0.50').reset_index(drop=True)
#plt.plot(test_results.reduced_data, test_results['qgb 0.50'], 'r+',label='median')
outliers = test_results.loc[(test_results.truth>test_results['qgb_t 0.95'])|(test_results.truth<test_results['qgb_t 0.05'])]

notoutlier = test_results.loc[(test_results.truth<test_results['qgb_t 0.95'])&(test_results.truth>test_results['qgb_t 0.05'])]

plt.scatter(outliers['qgb_t 0.50'], outliers['truth'],s=15,linewidths=1,color='red',marker='x',label='out of range')

plt.scatter(notoutlier['qgb_t 0.50'], notoutlier['truth'],s=15,linewidths=1,color='blue',marker='+',label='in range')
#sns.histplot(x=test_results['qgb_t 0.50'], y=test_results['truth'], bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=test_results['qgb_t 0.50'], y=test_results['truth'],  color="cyan", linewidths=1)
plt.fill_between(
test_results['qgb_t 0.50'].ravel(), test_results['qgb_t 0.05'], test_results['qgb_t 0.95'], alpha=0.4, label="Predicted 90% CI"
)

#sns.kdeplot(x=test_results['qgb_t 0.50'], y=test_results['truth'], levels=5, color="red", linewidths=1)
plt.xlabel('predicted median RMSE')
plt.ylabel('RMSE')
plt.xlim(0,test_results['qgb_t 0.50'].max())
plt.ylim(0,test_results['truth'].max()+5)
plt.plot([0,100],[0,100],'k--')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.grid(True)
plt.legend()

plt.show()
#%%
fig.savefig("../output_figures/f06.pdf",
       bbox_inches='tight',dpi=600,transparent=False,facecolor='white')