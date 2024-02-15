#%%
import scipy.stats
import pandas as pd

df = pd.read_feather('../output/train_test2')
df = df.dropna()
df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)

bad = df.loc[(df.mrms_lat==40.57499999999929)&(df.mrms_lon==254.91499899999639)]

# Example data (replace this with your own data)
population_mean = df.norm_diff.mean()  # replace with the actual population mean
sample_data = bad.norm_diff  # replace with your sample data

# Perform a one-sample t-test
t_statistic, p_value = scipy.stats.ttest_1samp(sample_data, population_mean)

# Print the results
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Compare p-value to the significance level
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The mean is significantly higher.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to conclude the mean is significantly higher.")

# %%
