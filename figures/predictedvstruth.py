#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def predictedvstruth(test_results, plot_name,fig_idx):
       if 'nr_rmse' in plot_name:
              ax_name2 = 'normalized RMSE'
              ax_name3 = 'predicted nRMSE'
              
       elif 'mean_error' in plot_name:
              ax_name2 = 'mean error'
              ax_name3 = 'predicted median mean error'

       else:
              ax_name2 = 'RMSE'
              ax_name3 = 'predicted median RMSE'
       fig = plt.figure(figsize=(10*.6, 8*.6))
       #test_results = test_results.divide(test.max_mrms.values,axis=0)
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
       plt.xlabel(ax_name3)
       plt.ylabel(ax_name2)
       plt.xlim(-5,20)
       plt.ylim(test_results['truth'].min(),test_results['truth'].max())
       plt.plot([0,100],[0,100],'k--')
       plt.gca().spines['top'].set_visible(False)
       plt.gca().spines['right'].set_visible(False)

       plt.grid(True)
       plt.legend()
       #plt.title(plot_name)
       plt.show()
       
       #fig.savefig('../output_figures/f06.pdf',bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
       fig.savefig("../output_figures/experiments/S"+str(fig_idx)+".pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')