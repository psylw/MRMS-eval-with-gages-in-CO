import pandas as pd

#import pandas as pd
compare = pd.read_feather('window_values_FN')

# open window values
accumulation_threshold = 1
compare = compare.loc[(compare.total_accum_atgage>accumulation_threshold)|(compare.total_gage_accum>accumulation_threshold)]

compare['onoff'] = 0
compare.loc[(compare.total_accum_atgage>0)&(compare.total_gage_accum>0),['onoff']]='TP'
compare.loc[(compare.total_accum_atgage==0)&(compare.total_gage_accum>0),['onoff']]='FN'
compare.loc[(compare.total_accum_atgage>0)&(compare.total_gage_accum==0),['onoff']]='FP'

compare.groupby('onoff').count()/len(compare)