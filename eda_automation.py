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


def plot_hist_basic(df, col):
    '''
    Return a Matplotlib axis object with a histogram of the data in col.

    Plots a histogram from the column col of dataframe df.

    Parameters
    ----------
    df: Pandas DataFrame

    col: str
        Column from df with numeric data to be plotted

    Returns
    -------
    ax: Matplotlib axis object
    '''
    data = df[col]
    ax = data.hist(bins=30, normed=1, edgecolor='none', figsize=(10, 7), alpha=.5)
    ax.set_ylabel('Probability Density')
    ax.set_title(col)

    return ax


class method_of_moments(object):
    """Cacluates and plots gamma and normal model method of moments estimates"""

    def __init__(self):
        """Construct methond_of_moments class"""


    def fit(self, df, col):
        """Fit Normal and Gamma models to the data using Method of Moments

        Parameters
        ----------
        df: Pandas DataFrame

        col: str
             Column from df with numeric data for Method of Moments
             distribution estimation and plotting
        """

        self.df = df
        self.col = col
        self.samp_mean = df[col].mean()
        self.samp_var = df[col].var(ddof=1)
        self._fit_gamma()
        self._fit_normal()

    def _fit_gamma(self):
        """Fit Normal and Gamma models to the data using Method of Moments"""
        self.alpha = self.samp_mean**2 / self.samp_var
        self.beta = self.samp_mean / self.samp_var

    def _fit_normal(self):
        """Fit Normal and Gamma models to the data using Method of Moments"""
        self.samp_std = self.samp_var**0.5



    def plot_pdf(self, plot_df=None, ax=None, gamma=True, normal=True, xlim=None, ylim=None, mask = None):
        """Plot distribution PDFs using MOM against histogram of data in df[col].

        Parameters
        ----------
        ax: Matplotlib axis object, optional (default=None)
            Used for creating multiple subplots

        gamma: boolean, optional (default=True)
               Fit and plot a Gamma Distribution

        normal: boolean, optional (default=True)
                Fit and plot a Normal Distribution

        xlim: None, or two element tuple
              If not 'None', these limits are used for the x-axis

        ylim: None, or two element tuple
              If not 'None', these limits are used for the y-axis

        mask: limitation on column data
        Returns
        -------
        ax: Matplotlib axis object
        """

        if plot_df is None:
            plot_df = self.df

        if ax is None:
            ax = plot_hist_basic(plot_df, self.col)

        x_vals = np.linspace(plot_df[self.col].min(), plot_df[self.col].max())

        if gamma:
            gamma_rv = stats.gamma(a=self.alpha, scale=1/self.beta)
            gamma_p = gamma_rv.pdf(x_vals)
            ax.plot(x_vals, gamma_p, color='b', label='Gamma MOM', alpha=0.6)

        if normal:
            # scipy's scale parameter is standard dev.
            normal_rv = stats.norm(loc=self.samp_mean, scale=self.samp_std)
            normal_p = normal_rv.pdf(x_vals)
            ax.plot(x_vals, normal_p, color='k', label='Normal MOM', alpha=0.6)

        ax.set_ylabel('Probability Density')
        ax.legend()

        if not xlim is None:
            ax.set_xlim(*xlim)

        if not ylim is None:
            ax.set_ylim(*ylim)

        return ax



class panda_eda(object):
    def __init__(self,df):
        self.df = df
        self.step = 0
        self.target_column = None
        self.numeric_columns = []
        self.feature_pipeline = []
        self.exploration_steps = []
        self.correlations = None
        #self.column_convert_date()

    def _go(self,pipe,label,code):
        '''
        execute and record executable code
        pipe: pipeline to add step to (self.feature_pipeline, presentation, model)
        label: str
        code: executable object
        '''
        pipe.append([label,code])
        exec(code)

    def save_feature_pipe(self):
        fp = pd.DataFrame(self.feature_pipeline,columns=['label','code'])
        fp.to_csv('data/{}_feature_list'.format(self.target_column))

    def p_groupby(self):
        self.df_str = self.df.select_dtypes(include=[np.object])
        self.groupings = []
        for col in self.df_str.columns:
            df_group = self.df.groupby(col)
            df_group.sum()
            df_group.mean()
            groupings.append((col,df_group))
        return groupings

    def create_bins(self,new_column,cut_column,range_list):
        self.labels = [ "{0} - {1}".format(i, i + 9) for i in range_list ]
        df[new_column] = pd.cut(self.df[cut_column], range(0, 105, 10), right=False, labels=labels)
        return

    def column_convert_date(self):
        '''automatically tries to select columns that looks like dates and converts them'''
        self.df = self.df.apply(lambda col: pd.to_datetime(col, errors='ignore')
              if col.dtypes == object
              else col,
              axis=0)
        return

    def ask_column_convert_date(self,pipe,label):
        a = input("Convert any date columns to type=datetime? (y/n)")
        if a == 'y': self._go(self.feature_pipeline,label,"self.column_convert_date()")

    def add_date_columns(self, column_name=None, date_desc=None):
        if column_name is None or len(column_name) == 0:
            column_name = df_num.select_dtypes(include=[np.datetime64]).columns
        if data_desc is None or len(data_desc)==0:
            data_desc = ['year','month','day','dayofweek','weekday_name','hour','time','minute']
        for dt_type in date_desc:
            self.df[self.column_name + '_' + dt_type] = pd.DatetimeIndex(self.df[self.column_name]).year

        if 1 ==2:
            self.df[self.column_name + '_year'] = pd.DatetimeIndex(self.df[self.column_name]).year
            self.df[self.column_name + '_month'] = pd.DatetimeIndex(df[self.column_name]).month
            self.df[self.column_name + '_day'] = pd.DatetimeIndex(df[self.column_name]).day
            self.df[self.column_name + '_hour'] = pd.DatetimeIndex(df[self.column_name]).hour
            self.df[self.column_name + '_dayofweek'] = pd.DatetimeIndex(df[self.column_name]).dayofweek
            self.df[self.column_name + '_weekday'] = pd.DatetimeIndex(df[self.column_name]).weekday_name
            self.df[self.column_name + '_time'] = pd.DatetimeIndex(df[self.column_name]).time
            self.df[self.column_name + '_minute'] = pd.DatetimeIndex(df[self.column_name]).minute
        return

    def ask_add_date_columns(self,pipe,label):
        a = input("Add date descriptions(day,month,year,dayofweek)? (y/n)")
        if a == 'y':
            column_names, date_desc = Input("What columns and data descriptors?/nExpecting: [column names; empty list=all datatime columns],[date desc,empty list=all]")
            self._go(pipe,label, "column_convert_date(column_names, date_desc)")

    def identify_index_column(self,pipe=None,label=None):
        '''
        tests whether a column appears to be a copy of the index and offer to drop it
        '''
        test_match = [(column,sum(eda.df[column]!=eda.df.index.tolist())) for column in eda.numeric_columns]
        match_column = [x for x,y in test_match if y == 0]
        if len(match_column)==0:
            test_match = [(column,sum(eda.df[column]-1!=eda.df.index.tolist())) for column in eda.numeric_columns]
            match_column = [x for x,y in test_match if y == 0]
        if len(match_column)>0:
            a = input("{} appears to be an index column.  Do you want to drop it?  (y/n) ".format(match_column) )
            if a=='y':
                self._go(self.feature_pipeline,'Drop index column',"self.drop_columns({})".format(match_column))
                print("{} was dropped".format(match_column))
        else: print("No index columns\n")

    def drop_columns(self,columns_to_drop,by_name=True):
        '''
        by_name:            bool
        columns_to_drop:    by_name=True:   list of column names if
                            by_name=False:   list of column index positions

        '''
        if by_name == False:
            self.df.drop(self.df.columns[columns_to_drop],axis=1,inplace=True) #pass an index to drop
        if by_name == True:
            self.df.drop(columns_to_drop,axis=1,inplace=True)

    def ask_drop_columns(self,pipe,label):
        a = input("Drop any columns? (y/n)")
        if a == 'y':
            column_to_drop, by_name = input("What columns to drop?/nExpecting: [column names or index],[by_name=True]")
            self._go(self.feature_pipeline,"drop_columns_ex", "drop_columns(column_to_drop, by_name)")

    def ask_set_target_column(self,pipe,label):
        if self.target_column is None:
            a = str(input("What is the target column?   "))
            while a not in self.df.columns:
                a = str(input("{} not in column list.\nColumn list: {}\nWhat is the target column?  ".format(a,self.df.columns.values)))
            self._go(pipe,label,"self.set_target_column('{}')".format(a))
        print('Target column set to {}.'.format(self.target_column))

    def set_target_column(self,target_column):
        self.target_columns = target_column

    def set_numeric_column(self,pipe=None,label=None):
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            print("Below are the numeric columns.\n{}".format(self.numeric_columns))

    def _escape_step(self):
        self.step+=1
        a = input("Press enter.  (q to quit exploration')  ")
        if a == 'q':
            print("You are at exploration step {}".format(self.step))
            return 'q'
        return

    def plot_target_hist(self):
        tar_chart = method_of_moments()
        tar_chart.fit(df=eda.df,col=eda.target_column)
        tar_chart.plot_pdf()

    def basic_panda(self):
        '''
        '''
        print('df.head()\n',self.df.head())
        a = input("press enter")
        print('df.info()\n',self.df.info())
        a = input("press enter")
        print('df.describe()\n',self.df.describe())
        return

    def null_analysis(self):
        pass

    def plot_scatter(self,lable=None):
        _ = scatter_matrix(self.df[self.numeric_columns], alpha=0.2, diagonal='kde')

    def get_corrs_dict(df):
        col_correlations = df.corr()
        col_correlations.loc[:, :] = np.tril(col_correlations, k=-1)
        cor_pairs = col_correlations.stack()
        return cor_pairs.to_dict()

    def get_corrs(self):
        for col_x in eda.numeric_columns:
            for col_y in eda.numeric_columns:
                corr = "{0:.2f}".format(eda.df[col_x].corr(eda.df[col_yx]))
                self.correlations.append(col_x,col_y,corr)

    def corr_with_target(self,pipe=None,label=None):
        print(self.correlations[self.correlations[col_x]==self.target_column].sort_values(by='corr',ascending=False))

    def high_covariance(self,pipe=None,label=None):
        print(self.correlations[(self.correlations[col_x]!=self.target_column) & (self.correlations[corr]>0.6)])

    def value_to_float(self,x):
        if type(x) == float or type(x) == int:
            return x
        if 'K' in x:
            if len(x) > 1:
                return float(x.replace('K', '')) * 1000
            return 1000.0
        if 'M' in x:
            if len(x) > 1:
                return float(x.replace('M', '')) * 1000000
            return 1000000.0
        if 'B' in x:
            return float(x.replace('B', '')) * 1000000000
        return 0.0

    def set_number_correct(self,column_name):
        self.df[column_name] = self.df[column_name].apply(self.value_to_float)

    def exploration_report(self,target_step=None,by_name=False):
        '''
        by_name:        bool
        target_step:    by_name=True:   request step by name
                        by_name=False:  request step by index position

        iterates thru options in logical order, taking answers and executing action or saving result
        should add named steps to a pipeline, so they can be called by step name or index, and easily re-ordered
        pipeline might also help keep track of answers, and to start where you left off if you explore
        '''

#        self.exploration_steps.append((None,'basic_panda',self.basic_panda))
        self.exploration_steps.append((self.feature_pipeline,'convert_to_datetime',self.ask_column_convert_date))
        self.exploration_steps.append((self.feature_pipeline,'add_date_descriptions',self.ask_add_date_columns))
        self.exploration_steps.append((None,'ID_index_column',self.identify_index_column))
        self.exploration_steps.append((self.feature_pipeline,'set_numeric_columns',self.set_numeric_column))
#        self.exploration_steps.append((None,'scatter_matrix',self.plot_scatter))
        self.exploration_steps.append((self.feature_pipeline,'set_target',self.ask_set_target_column))

        self.exploration_steps.append((None,'plot_target_hist',self.plot_target_hist))
        self.exploration_steps.append((None,'Correlation_with_target',self.corr_with_target))
        self.exploration_steps.append((None,'High_Correlation',self.high_covariance))


        self.exploration_steps.append((None,'plot_all_hist',self.df.hist))

        #execution_steps.append(('scatter_matrix',scatter_matrix(self.df, alpha=0.2, diagonal='kde')))
        #univariate plots
        #histograms
        #correlation matrix

        self.execute_steps(self.exploration_steps,target_step)

    def execute_steps(self,step_holder,target_step=None):
        if target_step is None:
            target_step = self.step

        for stepper in step_holder[target_step:]:
            pipe,label,code = step_holder[self.step]
            print("/n{}/n".format(label))
            if pipe is None:
                code()
            else: code(pipe,label) #selects the step by index, then executable, then parens execute it
            if self._escape_step() == 'q': return
        if self.step>len(step_holder): self.step=0
        return

    def __self__(self):
        return self.df

def presentation_plots(column_name,min_level, max_level,group_by):
    eda.df[(eda.df[column_name]>min_level)&(eda.df[column_name]<=max_level)].groupby(group_by)[column_name].sum().plot(title='Sum of Adjusted Damage by year',label="{},min={}, max={}".format(column_name,min_level,max_level))
    plt.legend()
    # eda.df.groupby('year')[column_name].count().plot(title='Count of observed storms by year')
    # eda.df.groupby('year')[column_name].mean().plot(title='Mean of Adjusted Damage by year')
    # eda.df.groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')
    # eda.df[eda.df['column_name']<min_level].groupby('year')['column_name'].sum().plot(title='Sum of Adjusted Damage by year\nAdjusted Damage < 4000')
    # eda.df[(eda.df[column_name]>min_level)&(eda.df[column_name]<=max_level)].groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')
    # eda.df[(eda.df[column_name]>25000)&(eda.df[column_name]<=100000)].groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')
    # eda.df[(eda.df[column_name]>100000)&(eda.df[column_name]<=500000)].groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')
    # eda.df[(eda.df[column_name]>500000)&(eda.df[column_name]<=5000000)].groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')
    # eda.df[(eda.df[column_name]>5000000)&(eda.df[column_name]<=50000000)].groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')
    # eda.df[(eda.df[column_name]>=50000000)].groupby('year')[column_name].sum().plot(title='Sum of Adjusted Damage by year')


def plot_by_severity():
    pass

def presentation():
    eda.df.boxplot('log_adj_damage',by='decade')
    condition1= eda.df['log_adj_damage']>17
    eda.df[condition1].boxplot('log_adj_damage',by='decade',figsize=(15,15))
    eda.df.hist('log_adj_damage',by='decade',figsize=(20,20),bins=40,alpha=.2)
    eda.df[condition1].hist('log_adj_damage',by='decade',figsize=(20,20),bins=40,alpha=.2)
    eda.df[condition_severityA].boxplot('log_adj_damage',by='decade')

def interesting_code():
    eda.df['event_type'].unique()
    pd.crosstab(tornado_df['decade'],tornado_df['severity'])

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
    filename = 'data/storm_data.csv'
    df_master = pd.read_csv(str(path + filename))
    eda = panda_eda(df_master)
    eda.df.columns = [str.lower(column) for column in eda.df.columns]
    eda.set_number_correct('damage_crops')
    eda.set_number_correct('damage_property')
    eda.df['conv_f_scale'] =  eda.df['tor_f_scale'].map(tor_conversion)

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
    eda.df = eda.df[eda.df['adj_damage']>0]

    eda.set_numeric_column()
    eda.drop_columns(0,by_name=False)
    eda.target_column='adj_damage'
    eda.df.drop(['gdp_adj_factor','event_id'],axis=1,inplace=True)
    eda.df['log_adj_damage'] = np.log(eda.df['adj_damage'])
    range_list = range(1950,2021,10)
    yr_labels = [ "{0}s".format(i, i + 9) for i in range_list[:-1]]
    eda.df['decade'] = pd.cut(eda.df.index, range_list, right=False, labels=yr_labels)
    condition_50s = (eda.df['decade']=='1950s')

    event_targets = ['Tornado','TORNADOES']
    condition_tornadoes = eda.df['event_type'].isin(event_targets)

    severity_max = eda.df[condition_50s]['adj_damage'].max()
    severity_range = np.log([1,severity_max/1000,severity_max/100,severity_max/10,3*severity_max])
    severity_labels = ['A','B','C','D']
    eda.df['severity'] = pd.cut(eda.df['log_adj_damage'], severity_range, right=False, labels=severity_labels)
    condition_severityA = eda.df['severity']=='A'
