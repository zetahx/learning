import pandas as pd
import numpy as np
from pandas.core.generic import NDFrame
# n_data = 100
# df = pd.read_csv('test.csv')
# list_id = df['id'].head(n_data*100)
# list_ts = df['ts'].drop_duplicates().head(n_data).tolist()
# list_x = df['x'].drop_duplicates().head(n_data).tolist()
# list_y = df['y'].drop_duplicates().head(n_data).tolist()
# 
# new_ts = []
# new_x = []
# new_y = []
# for i in range(n_data):
#     new_ts +=  list_ts
#     new_x += ([float(list_x[i])]*n_data)
#     new_y += ([float(list_x[i])]*n_data)
# new_ts = pd.Series(new_ts)
# new_x = pd.Series(new_x)
# new_y = pd.Series(new_y)
#   
# ndf = pd.DataFrame({'ts':new_ts,'x':new_x,'y':new_y})
# # print (new_ts)
# ndf['id'] = list_id
# 
# 
# ndf['id']   = ndf['id'].iloc[1]+1
# 
# print (ndf)
# ndf.to_csv('data.csv',index = False)


