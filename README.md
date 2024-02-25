# MRMS-eval-with-gages-in-CO

We evaluate 8-hour time series samples of MRMS 15-minute intensity through a comparison to time series from 204 gages located in the mountains of Colorado. For each time series sample, various features related to the physical characteristics influencing MRMS performance are calculated from the topography, surrounding storms, and rainfall observed at the gage location. A gradient-boosting regressor is trained on these features and is optimized with quantile loss, using the RMSE as a target, to model nonlinear patterns in the features that relate to a range of error. This model was used to predict a range of error throughout the mountains of Colorado during warm months, spanning 6 years, resulting in a spatiotemporally varying error model of MRMS for sub-hourly precipitation rates.




