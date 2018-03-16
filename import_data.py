import pandas as pd
import numpy as np

df_master = pd.read_csv('data/NOAA_storm_files/StormEvents_details-ftp_v1.0_d1950_c20170120.csv')

files = pd.read_csv('csv_list')
files.columns=['filename']
files = files['filename'].str.replace('.gz','')
my_cols = ['STATE','YEAR','MONTH_NAME','EVENT_TYPE','EVENT_ID','BEGIN_DATE_TIME','END_DATE_TIME','DAMAGE_PROPERTY','DAMAGE_CROPS','DEATHS_DIRECT','DEATHS_INDIRECT','INJURIES_DIRECT','INJURIES_INDIRECT','MAGNITUDE','BEGIN_LAT','BEGIN_LON','TOR_F_SCALE']
df_master = df_master[my_cols]
for i,file in enumerate(files.values):

    if i>=34 and file !='StormEvents_details-ftp_v1.0_d1950_c20170120.csv':
        print("loading {}".format(file))
        df = pd.read_csv('data/NOAA_storm_files/{}'.format(file))
        df = df[my_cols]
        df_master = pd.concat([df_master,df])


df_master.to_csv('data/storm_data_86.csv')
'''
convert DAMAGE codes to dollars - K to 1000, M to million
can't think right now.  maybe apply a lambda, or extact letter and multiply

maybe injuries or death are predictive... in total or a few sever incidence
again, looking for reactivity


StormEvents_details-ftp_v1.0_d1996_c20170717.csv.gz
StormEvents_details-ftp_v1.0_d1997_c20170717.csv.gz
StormEvents_details-ftp_v1.0_d1998_c20170717.csv.gz
'''
