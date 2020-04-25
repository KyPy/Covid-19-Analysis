# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:21:50 2020

@author: kaisa
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

sns.set()


#  Import data 

# Covid-19 time series
covid_conf = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
covid_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

# transform data
def transform_covid_data(df):
    
    df.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
    df = df.groupby('Country/Region').sum()
    df = df.transpose()
    df.index = [datetime.strptime(d, '%m/%d/%y') for d in df.index]
    
    return df

covid_conf = transform_covid_data(covid_conf)
covid_deaths = transform_covid_data(covid_deaths)



def get_time_series(df, country, min_value):
    s = df.loc[df[country]>=min_value, country]
    s.index = np.array([datetime.timestamp(x) for x in s.index])/(3600*24)
    s.index -= s.index[0]
    return s


countries = ['India', 'Korea, South', 'Italy', 'Germany', 'US', 'France']

for i, c in enumerate(countries):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(10,4.5))
    
    s = get_time_series(covid_conf, c, 100)
    
    color = cm.viridis(100)
    ax[0].plot(s, color=color, label='Cases (total)')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].set_xlabel('Days since 100 cases')
    
    ax2 = ax[0].twinx()
    
    color = cm.viridis(150)
    ax2.plot(s.index, np.gradient(s, s.index), color=color, label='Cases per day')
    ax2.tick_params(axis='y', labelcolor=color)
    
    
    lines, labels = ax[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.grid(None)
    
    
    
    s = get_time_series(covid_deaths, c, 10)
    
    color = cm.viridis(100)
    ax[1].plot(s, color=color, label='Deaths (total)')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].set_xlabel('Days since 10 deaths')
    
    ax2 = ax[1].twinx()
    
    color = cm.viridis(150)
    ax2.plot(s.index, np.gradient(s, s.index), color=color, label='Deaths per day')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.grid(None)
    
    plt.suptitle(c)
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\diagramme',c+'_series'), dpi=200)



