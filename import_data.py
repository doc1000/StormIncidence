import pandas as pd
import numpy as np

df_master = pd.read_csv('data/StormEvents_details-ftp_v1.0_d1950_c20170120.csv')

files = pd.read_csv('csv_list')
files.columns=['filename']
files = files['filename'].str.replace('.gz','')
my_cols = ['STATE','YEAR','MONTH_NAME','EVENT_TYPE','EVENT_ID','DAMAGE_PROPERTY','DAMAGE_CROPS','DEATHS_DIRECT','DEATHS_INDIRECT','INJURIES_DIRECT','INJURIES_INDIRECT','MAGNITUDE','BEGIN_LAT','BEGIN_LON','DATA_SOURCE']
df_master = df_master[my_cols]
for i,file in enumerate(files.values):

    if i<70 and file !='StormEvents_details-ftp_v1.0_d1950_c20170120.csv':
        print("loading {}".format(file))
        df = pd.read_csv('data/{}'.format(file))
        df = df[my_cols]
        df_master = pd.concat([df_master,df])


pd.df_master.to_csv('/data/storm_data.csv')
'''
convert DAMAGE codes to dollars - K to 1000, M to million
can't think right now.  maybe apply a lambda, or extact letter and multiply

maybe injuries or death are predictive... in total or a few sever incidence
again, looking for reactivity

'''
