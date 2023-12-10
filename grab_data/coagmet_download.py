# download coagmet warm months, one year at a time
# %%

import os
import urllib.request


destination = os.path.join('..', 'output')

for year in range(2018,2024):

    filename = str(year)+'_coagmet'

    data = 'https://coagmet.colostate.edu/data/5min.csv?header=yes&from='+str(year)+'-05-01&to='+str(year)+'-09-30&tz=utc&units=m&fields=precip'

    fetched_request = urllib.request.urlopen(data)
    
    with open(destination + os.sep + filename, 'wb') as f:
        f.write(fetched_request.read())