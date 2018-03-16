import matplotlib.pyplot as plt
import seaborn
import numpy as np
#%matplotlib inline
#ipython --pylab #keeps graphics open
import seaborn as sns
import pandas as pd
#import pandas_profiling as pp
from pandas.plotting import scatter_matrix

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression, LogisticRegression

#this would require a special install, could code that as well I suppose
# from regression_tools.plotting_tools import (
#     plot_univariate_smooth,
#     bootstrap_train,
#     display_coef,
#     plot_bootstrap_coefs,
#     plot_partial_depenence,
#     plot_partial_dependences,
#     predicteds_vs_actuals)



from eda_automation import panda_eda,method_of_moments, plot_hist_basic




if __name__ == '__main__':
    # path = '~/galvanize/ds-case-study-linear-models/forecast_HIV_infections/'
    # filename = 'all_merged_HIV.csv'
    # HIV = pd.read_csv(str(path + filename))
    #
    # eda = panda_eda(HIV)
    # eda.target_column='HIVincidence'
    # #tar_chart = method_of_moments()
    #tar_chart.fit(df=eda.df,col=eda.target_column)
    path = ''
    filename = 'data/storm_data_85.csv'
    df_master = pd.read_csv(str(path + filename))
    eda = panda_eda(df_master)

    eda.drop_columns(0,by_name=False)

    eda.df.columns = [str.lower(column) for column in eda.df.columns]
    eda.set_number_correct('damage_crops')
    eda.set_number_correct('damage_property')
    tor_conversion = {'EF0':'F0','EF1':'F1','EF2':'F2','EF3':'F3','EF4':'F4','EF5':'F5','F0':'F0','F1':'F1','F2':'F2','F3':'F3','F4':'F4','F5':'F5'}
    eda.df['conv_f_scale'] =  eda.df['tor_f_scale'].map(tor_conversion)
    magnitude_range = [40,73,113,158,207,261,318]
    magnitude_labels = ['F0','F1','F2','F3','F4','F5']
    eda.df['mag_f_scale'] = pd.cut(eda.df['magnitude'],magnitude_range,right=False,labels=magnitude_labels)
    eda.df['conv_f_scale'] = np.where(eda.df['conv_f_scale'].notnull(),eda.df['conv_f_scale'],eda.df['mag_f_scale'])

    eda.df = eda.df[eda.df['conv_f_scale'].notnull()]
    #event_targets = ['Tornado','TORNADOES','TORNADOES, TSTM WIND, HAIL']
    #condition_tornadoes = eda.df['event_type'].isin(event_targets)
    #eda.df = eda.df[eda.df['event_type'].isin(event_targets)]
    eda.df = eda.df[eda.df['event_type']!='Hail']
    eda.df = eda.df[eda.df['event_type']!='HAIL FLOODING']
    #eda.column_convert_date()

    '''
    if there isn't property damage, prob don't want to look at it
    however, there are some with crop damage and with injuries and deaths, that may be useful
    '''
    #eda.df = eda.df[eda.df['DAMAGE_PROPERTY']>0]
    #import GDP data to normalize with
    GDP = pd.read_csv('data/GDP.csv')
    GDP['year'] = pd.DatetimeIndex(GDP['DATE']).year
    GDP_year = GDP.groupby('year').mean()
    base_year_gdp =GDP_year[GDP_year.index==2017]['GDP'].values[0]
    GDP_year['gdp_adj_factor']=GDP_year['GDP']/base_year_gdp
    eda.df = eda.df.set_index('year').join(GDP_year['gdp_adj_factor'])
    eda.df['adj_damage'] = (eda.df['damage_property']+eda.df['damage_crops'])/eda.df['gdp_adj_factor']
    #eda.df = eda.df[eda.df['adj_damage']>0]
    eda.df['begin_date'] = pd.DatetimeIndex(eda.df['begin_date_time']).date
    eda.df['end_date'] = pd.DatetimeIndex(eda.df['end_date_time']).date

    eda.df = eda.df.groupby(['year','state','begin_date','end_date','event_type'])[['adj_damage','damage_property','conv_f_scale','deaths_direct','deaths_indirect','injuries_direct','injuries_indirect']].max().reset_index()
    #eda.df = eda.df.to_frame()
    #eda.df = eda.df.reset_index(level='year')
    #eda = panda_eda(df)
    #del df
    eda.set_numeric_column()
    eda.target_column='adj_damage'
    #eda.df.drop(['gdp_adj_factor'],axis=1,inplace=True)
    eda.df['log_adj_damage'] = np.log(eda.df['adj_damage']+1)
    #range_list = range(1950,2021,10)
    range_list = range(1958,2019,10)

    yr_labels = [ "{0}-{1}".format(i, i + 9) for i in range_list[:-1]]
    eda.df['decade'] = pd.cut(eda.df['year'], range_list, right=False, labels=yr_labels)
    condition_50s = (eda.df['decade']=='1958-1967')


    severity_max = eda.df[condition_50s]['adj_damage'].max()
    severity_range = np.log([0.5,severity_max/1000,severity_max/100,severity_max/10,3*severity_max])
    severity_labels = ['A','B','C','D']
    eda.df['severity'] = pd.cut(eda.df['log_adj_damage'], severity_range, right=False, labels=severity_labels)
    condition_severityA = eda.df['severity']=='A'


    #early_tor_rate = eda.df[eda.df['year']<1984]['state'].count()/33
    #late_tor_rate = eda.df[eda.df['year']>=1984]['state'].count()/34

    eda.df.to_csv('data/f_scale_storm_data_85.csv')
