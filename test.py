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

from bokeh.plotting import ColumnDataSource, figure, output_file, show, save
from bokeh.models import HoverTool

sns.set()


# -----------------    Import data   ---------------------

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

# Country data
country_info = pd.read_csv(os.path.join('.', 'data', 'covid19countryinfo.csv'))
def convert_string(x):
    try:
        return np.float(x.replace(',',''))
    except:
        return np.nan
    
for c in ['pop', 'gdp2019', 'healthexp']:
    country_info[c] = country_info[c].apply(convert_string)

# Restrictions
restrictions = pd.read_csv(os.path.join('.', 'data', 'restrictions.csv'),sep=';')
restrictions['date'] = pd.to_datetime(restrictions['date'], format='%d.%m.%Y')


# ---------   Question 1: Duration till turning point   ----------

def get_time_series(df, country, min_value):
    s = df.loc[df[country]>=min_value, country]
    s.index = np.array([datetime.timestamp(x) for x in s.index])/(3600*24)
    s.index -= s.index[0]
    return s

"""
countries = ['China', 'Korea, South', 'Italy', 'Germany', 'US']

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
"""


# ----------   Question 2: Effect of restrictions   ------------


def add_annotations(ax, df, s):
    
    last_y = 0
    
    df.reset_index(drop=True, inplace=True)
    for i, row in df.iterrows():
        
        y = s.iloc[s.index.get_loc(row.date, method='nearest')]
        x_text = row.date - timedelta(days=10)
        y_text = y + s.max()/10
        y_text = max(y_text, last_y+s.max()/12)
        last_y = y_text
        
        ann = ax.annotate(str(i+1),
                          xy=(row.date, y), xycoords='data',
                          xytext=(x_text, y_text), textcoords='data',
                          size=15, va="center", ha="center",
                          bbox=dict(boxstyle="round4", fc="w"),
                          arrowprops=dict(arrowstyle="-|>",
                                          connectionstyle="arc3,rad=-0.2",
                                          fc="k", color='k'),
                          )
        
        plt.text(1.02, 0.92-i*0.06, '{:d}: {}'.format(i+1,row.text), horizontalalignment='left',
              verticalalignment='top', transform=ax.transAxes,
              fontsize=11)
    plt.text(1.02, 1, 'Restrictions / Actions:', horizontalalignment='left',
              verticalalignment='top', transform=ax.transAxes,
              fontsize=13, fontweight='bold')


countries = ['Korea, South', 'Italy', 'Germany']

for i, c in enumerate(countries):
    
    fig, ax = plt.subplots(figsize=(9,4))
    s = covid_conf[c]
    plt.plot(s)
    
    ax.set_xlim((s.idxmin(),s.idxmax()+timedelta(days=5)))
    myFmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(myFmt)
    
    ax.set_ylabel('Confirmed cases (total)')
    
    fig.tight_layout()
    plt.subplots_adjust(right=0.6, top=0.93)
    
    plt.suptitle(c)
    
    add_annotations(ax, restrictions.loc[restrictions.country_region==c], s)
    plt.savefig(os.path.join('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\diagramme',c+'_measures'), dpi=200)




# ----------   Question 3: Correlation with death/cases ratio   ------------

from collections import defaultdict
from bokeh.palettes import Viridis

ratio = defaultdict(list)

df_death_ratio = []




country_info['death_ratio'] = np.nan

for c in covid_conf.columns:
    
    df = pd.concat([pd.Series(covid_conf[c], name='Cases'), pd.Series(covid_deaths[c], name='Deaths')],axis=1)
    
    df = df.loc[df.Deaths>50]
    if len(df) == 0:
        continue
    
    death_ratio = pd.Series(df.Deaths / df.Cases, name=c)
    country_info.loc[country_info.country==c,'death_ratio'] = death_ratio.iloc[-1]
    df_death_ratio.append(death_ratio)
    
    ratio['date'].append(death_ratio.index)
    ratio['death_ratio'].append(np.array(death_ratio))
    ratio['country'].append(c)
    #ratio['color'].append(Viridis)

for i in range(len(ratio['country'])):
    ratio['color'].append(Viridis[256][int(i/len(ratio['country'])*256)])


df_death_ratio = pd.concat(df_death_ratio, axis=1)
country_info.dropna(subset=['death_ratio', 'healthperpop'], inplace=True)

# drop very small countries
country_info = country_info.loc[country_info['pop']>1E6]


correlation_columns = ['pop', 'density', 'medianage', 'urbanpop', 
       'hospibed', 'smokers', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019',
       'healthexp', 'healthperpop']
correlation = [country_info['death_ratio'].corr(country_info[c]) for c in correlation_columns]

fig, ax = plt.subplots(figsize=(7,4))
plt.bar(range(len(correlation)), correlation)
plt.xticks(range(len(correlation)), correlation_columns, rotation=90)
plt.ylabel('Correlation with mortality')
fig.tight_layout()
plt.savefig(os.path.join('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\diagramme','correlation'), dpi=200)




#source = ColumnDataSource(ratio)
#
#TOOLTIPS = [("country", "@country")]
#
#p = figure(plot_width=600, plot_height=400, tooltips=TOOLTIPS,
#           title="Covid-19 death ratio over time", x_axis_type='datetime') 
#
#p.multi_line(xs='date', ys='death_ratio',
#             line_width=5, line_color='color', line_alpha=0.6,
#             hover_line_color='color', hover_line_alpha=1.0,
#             source=source)
#
#p.xaxis.axis_label = "Date"
#p.yaxis.axis_label = "Covid-19 deaths / confirmed cases"
#
##show(p)
#save(p, filename=os.path.join('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\diagramme','mortality'))
#
#source = ColumnDataSource(country_info)
#
#TOOLTIPS = [("country", "@country"),
#            ("Mortality", "@death_ratio")]
#
#p = figure(plot_width=600, plot_height=400, tooltips=TOOLTIPS,
#           title="Influence of hospital capacity") #, x_axis_type="log"
#
#p.circle('hospibed', 'death_ratio', size=10, source=source)
#
#p.xaxis.axis_label = "Hospital beds per 1000 people"
#p.yaxis.axis_label = "Covid-19 deaths / confirmed cases,\n as on {}".format(datetime.strftime(covid_conf.index[-1], "%Y-%m-%d"))
#
##show(p)
#save(p, filename=os.path.join('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\diagramme','mortality_beds.html'))
#
#
#p = figure(plot_width=600, plot_height=400, tooltips=TOOLTIPS,
#           title="Influence of health care expenses", x_axis_type="log") #
#
#p.circle('healthperpop', 'death_ratio', size=10, source=source)
#
#p.xaxis.axis_label = "Health care expenses per 1 Mio. people"
#p.yaxis.axis_label = "Covid-19 deaths / confirmed cases, as on {}".format(datetime.strftime(covid_conf.index[-1], "%Y-%m-%d"))
#
##show(p)
#save(p, filename=os.path.join('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\diagramme','mortality_expenses.html'))
