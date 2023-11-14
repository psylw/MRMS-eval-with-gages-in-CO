import random
import pandas as pd
import matplotlib.pyplot as plt

# open both window values
test = pd.read_feather('window_values_FN')

compare = pd.read_feather('window_values')

accumulation_threshold = 1
compare = compare.loc[(compare.total_accum_atgage>accumulation_threshold)|(compare.total_gage_accum>accumulation_threshold)]

######################   COMPARE VALUES    ###############################################################################################

random.randint(0, len(compare))
for i in range(100):
    i = random.randint(0, len(compare))
    
    c = compare.iloc[i]
    
    axs[0].scatter(c.gage,c.mrms)
    axs[0].set_title('radar only')
    
    t = test.loc[(test['index']==c['index'])&(test['gage_id']==c['gage_id'][0])]
    
    try:
        axs[1].scatter(t.gage.iloc[0],t.mrms.iloc[0])
        axs[1].set_title('with correction')
    except:
        pass

######################   COMPARE TIME SERIES    ###############################################################################################
for i in range(100):

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    c = compare.iloc[i]
    
    axs[0].plot(c.mrms,label='mrms')
    axs[0].plot(c.gage,label='gage')
    axs[0].legend()
    axs[0].set_title(str(compare.mce_unsorted.iloc[i])[0:4])
    
    t = test.loc[(test['index']==c['index'])&(test['gage_id']==c['gage_id'][0])]
    axs[1].plot(t.mrms.iloc[0],label='mrms')
    axs[1].plot(t.gage.iloc[0],label='gage')
    axs[1].legend()
    axs[1].set_title(str(test.mce_unsorted.iloc[i])[0:4])
    
    plt.show()