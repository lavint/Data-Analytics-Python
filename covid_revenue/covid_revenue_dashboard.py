# Import dependencies
from pathlib import Path
import pandas as pd
import datetime as datetimelib
from datetime import datetime

from bokeh.models import BasicTickFormatter
from bokeh.models import NumeralTickFormatter
from bokeh.models import HoverTool

import seaborn as sns    
import matplotlib.pyplot as plt

import holoviews as hv
import panel as pn
pn.extension('plotly')
import hvplot.pandas

import statsmodels.api as sm


# Read CSVs
df = pd.read_csv((Path("data/stores_revenue.csv")))
df_cal = pd.read_csv('data/calendar.csv')
df_covid = pd.read_csv(Path("data/covid.csv"))


# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'])
df['zip_code'] = df['zip_code'].astype('str')


# Create weekly revenue DataFrame groupby date and state
by_state = df[['date', 'youtube', 'tiktok', 'state']].copy().groupby(['date', 'state']).sum()
by_state.reset_index(inplace=True)


# Clean holiday data
cal = df_cal[df_cal['HOL_FLG']==1].copy()
cal['HOL_DATE'] = pd.to_datetime(cal['HOL_DATE'])
cal.drop_duplicates(subset='HOL_DATE', inplace=True)
cal = cal[cal['HOL_DATE'] < '2020-06-30']
holiday = cal[['HOL_DATE', 'HOL_DESCP']].copy()
holiday['date'] = holiday['HOL_DATE'].apply(lambda x: x + datetimelib.timedelta(days=-x.weekday()))


# Create weekly COVID DataFrame
df_covid['date'] = pd.to_datetime(df_covid['date'])
df_covid = df_covid.drop(columns=['onVentilatorCumulative', 'inIcuCumulative', 'recovered'])
df_covid_weekly = df_covid.groupby('state').resample('W-MON', on='date').sum().reset_index().sort_values(by='date').copy()


def holiday_scatter_plot(org):
    ''' Create a scatter plot for holiday'''
    
    df_org = df[['date', org]].copy().groupby(['date']).sum()
    df_org.reset_index(inplace=True)
    df_org_holiday = pd.merge(df_org, holiday, 'right', on='date') 

    hover = HoverTool(tooltips=[('Date', '@date{%F}'),
                                ('Holiday/Special Event', '@HOL_DESCP'),
                                (f'{org.upper()} Revenue', f'@{org}' + '{$0.00 a}')], 
                      formatters= {'@date':'datetime'}, 
                      mode='vline' )

    return df_org_holiday.hvplot.scatter(x='date', 
                                         y=org, 
                                         color='#8c10c9', 
                                         size=50, 
                                         hover_cols=['HOL_DESCP'], 
                                         width=1000, 
                                         height=400, 
                                         yformatter=NumeralTickFormatter(format="$0a")).opts(tools=[hover], axiswise=True)


def line_revenue_over_time():
    '''Create a line chart for weekly revenue sum over time'''
    
    df_revenue = df[['date', 'tiktok', 'youtube']].copy().groupby(['date']).sum()
    revenue_plot = df_revenue.hvplot.line(x='date',
                                    y=['tiktok', 'youtube'],
                                    yformatter=NumeralTickFormatter(format="$0a"),
                                    xlabel='Date',
                                    ylabel='Revenue in US dollars',
                                    width=1300,
                                    shared_axes=False,
                                    hover=False
                                   )
    
    return (holiday_scatter_plot('tiktok') * holiday_scatter_plot('youtube') * revenue_plot
           ).opts(title = 'Revenue over Time Highlighted Holidays/Special Events in Purple Dots')



def bar_revenue_per_state():
    '''Create a side-by-side bar chart for each state'''
    
    by_state_plot = by_state.groupby(['state']).sum().sort_values('youtube', ascending=False)
    hover = HoverTool(tooltips=[("State","@state"),("Org", "@Variable"),("revenue","@value{$0a}")])
    
    return by_state_plot.hvplot.bar(x='state', 
                         y=['tiktok', 'youtube'],
                         xlabel='State',
                         ylabel='Revenue in US dollars',
                         yformatter=NumeralTickFormatter(format="$0a"),
                         bar_width=1,
                         width=1000
                        ).opts(multi_level=False, 
                               bgcolor='lightgray',
                               tools=[hover], 
                               axiswise=True,
                               title = 'Total Revenue from 2019 Jan to 2020 Jun')



def line_revenue_per_state_over_time():
    '''Create a line chart for weekly revenue over time per state with drop down option'''

    return by_state.hvplot.line(x='date',
                            y=['tiktok', 'youtube'],
                            groupby='state',
                            xlabel='Date',
                            ylabel='Revenue in US dollars',
                            yformatter=NumeralTickFormatter(format='$0a'),
                            group_label='Org',
                            ).opts(axiswise=True, 
                                   title = 'Revenue by State over Time')



def revenue_yoy_per_month():
    '''Create a year-over-year line chart for monthly revenue'''
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df_by_year = df[['year', 'month', 'tiktok', 'youtube']]
    df_by_year = pd.pivot_table(df_by_year, index=df_by_year['month'], columns=df_by_year['year'], values=['tiktok', 'youtube'], aggfunc='sum')
    df_by_year.style.format("{:1f}")
    
    yrcol = df_by_year.columns

    df_by_year.columns = df_by_year.columns.set_levels([yrcol.levels[0], yrcol.levels[1].astype(str)])
    df_by_year.columns = df_by_year.columns.map('_'.join)

    return df_by_year.hvplot.line(line_color=['lightblue', 'steelblue', 'peachpuff', 'orange'],
                           yformatter=NumeralTickFormatter(format="$0a"),
                           xlabel='Month',
                           ylabel='revenue in US dollars',
                           group_label='org_year',
                           width=1000, 
                           height=400,
                           shared_axes=False
                          ).opts(axiswise=True, 
                                 title='Tiktok & Youtube - 2019 & 2020 Revenue by Month')


def covid():
    '''Create a side-by-side bar charts to show number of cases for different COVID categories'''
    
    df_covid_weekly = df_covid.groupby('state').resample('W-MON', on='date').sum().reset_index().sort_values(by='date').copy()
    df_covid_weekly['month'] = df_covid_weekly['date'].dt.month
    df_covid_weekly = df_covid_weekly[df_covid_weekly['month'] > 2]
    df_covid_weekly_bar = df_covid_weekly[['month', 'deathIncrease', 'positiveIncrease', 'recoveredIncrease', 'hospitalizedIncrease']]
    covid_weekly_bar_plot = df_covid_weekly_bar.groupby('month').sum()
    
    return covid_weekly_bar_plot.hvplot.bar(rot=75, 
                                            width=1000, 
                                            height=400, 
                                            xlabel='Month, Status',
                                            ylabel='Number of cases', 
                                            yformatter=NumeralTickFormatter(format="0.0a"),
                                            title='Covid Cases by Month',
                                            shared_axes=False)



def revenue_trend(org, color):
    '''Create a line chart to show weekly revenue trend'''
    
    df_model = df[['date', 'tiktok', 'youtube']].groupby('date').sum()
    _, trend = sm.tsa.filters.hpfilter(df_model[org])
    
    return trend.hvplot(yformatter=NumeralTickFormatter(format="0.0a"),
                        color=color
                       ).opts(title=f'{org.upper()} Trend')


def revenue_noise_std(org, color):
    '''Create a line chart to show weekly revenue noise standard deviation'''
    
    df_model = df[['date', 'tiktok', 'youtube']].groupby('date').sum()
    noise, _ = sm.tsa.filters.hpfilter(df_model[org])
    
    return noise.hvplot(), hv.HLine(noise.sum() + noise.std()).opts(
                                                    color=color, 
                                                    line_dash='dashed', 
                                                    line_width=2.0,
                                                    yformatter=NumeralTickFormatter(format="0.0a"),
                                                    shared_axes=False
                        ), hv.HLine(noise.sum() - noise.std()).opts(
                                                    color=color, 
                                                    line_dash='dashed', 
                                                    line_width=2.0,
                                                    shared_axes=False
                        )

        
def revenue_noise(org, color):
    '''Create a line chart to show weekly revenue noise'''
    
    return (revenue_noise_std(org, color)[0]* revenue_noise_std(org, color)[1] * revenue_noise_std(org, color)[2])



def revenue_covid_correlation():
    '''Create a heatmap to show correlation of weekly revenue and covid categories'''
    
    df_store_revenue_state = df.groupby(['date', 'state']).sum().reset_index()
    df_revenue_covid = df_store_revenue_state.merge(df_covid_weekly, on=['date', 'state'])
    df_revenue_covid.drop(columns=['year', 'month'], inplace=True)
    fig = plt.figure(figsize=(15,15))
    correlation = df_revenue_covid.corr()
    ax = plt.axes()
    sns.heatmap(correlation, annot=True)
    ax.set_title('Tiktok & Youtube Weekly Revenue among 14 States and COVID Cases Correlation', fontsize = 15)
    plt.close(fig)
    
    return(fig)


# Create a title for the Dashboard
title = pn.pane.Markdown(
    """
    # Tiktok and Youtube Revenue Analysis from 2019 Jan to 2020 Jun
    """,
    width=800,
)

# Create markdown for Welcome tab description
welcome = pn.pane.Markdown(
    """
    This dashboard presents a visual analysis of Tiktok and Youtube revenue from 2019 to 2020.
    You can navigate through the tabs to explore more details.
    """
)

# Create markdown for Revenue at a glance tab description
tab2 = pn.pane.Markdown(
    """
    This tab shows revenue over time. 
    """
)

# Create markdown for Revenue Trend tab description
tab3 = pn.pane.Markdown(
    """
    This tab shows revenue noise and trends. 
    """
)

# Create markdown for Covid Revenue Correlation tab description
tab4 = pn.pane.Markdown(
    """
    This tab shows revenue and COVID cases correlation. 
    """
)

# Create a tab layout for the dashboard
tabs = pn.Tabs(
                ("Welcome", pn.Column(welcome, "<br>", revenue_yoy_per_month(), "<br><br>", covid())),
                ("Revenue at a glance", pn.Column(tab2, "<br>", line_revenue_over_time(), "<br>", 
                                             bar_revenue_per_state(), "<br>", line_revenue_per_state_over_time())),
                ("Revenue Trend", pn.Column(tab3, "<br>", (revenue_noise('tiktok', 'lightblue') * revenue_noise('youtube', 'peachpuff')
                                                     ).opts(title='Tiktok vs Youtube Noise', width=1300, height=500, shared_axes=False),
                                       revenue_trend('tiktok', 'steelblue'), revenue_trend('youtube', 'orange'))),
                ("Covid Revenue Correlation", pn.Column(tab4, revenue_covid_correlation()))
              )


# Create the dashboard
dashboard = pn.Column(pn.Row(title), tabs, width=900)

# Execute Panel dashboard using servable function
dashboard.servable()

### In Anaconda Prompt, navigate to the folder that has covid_revenue.py and type below
### panel serve --show covid_revenue.py