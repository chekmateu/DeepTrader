import polars as pl
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import RangeTool, HoverTool, ColumnDataSource, CDSView, IndexFilter
from math import pi
import statsmodels.tsa.stattools as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

'''
Data: ohlcv data -> pd.Dataframe
cswidth: 10*60*60
'''
def FinancePlot(data, plot = True, cswidth = 12*60*60, log = False):

    if log:
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].apply(np.log10)
        
    padding = 0.1 * (np.max(data['high']) - np.min(data['low']))
    yrange = [np.min(data['low'])  - padding, np.max(data['high']) + padding]

    source = ColumnDataSource(data = data)

    TOOLS = "xpan,wheel_zoom,reset,save"

    hover_tool = HoverTool(
        tooltips = [
            ('Date', '@{time}'),
            ('Open', '@{open}{%0.2f}'),
            ('Close', '@{close}{%0.2f}'),
            ('High', '@{high}{%0.2f}'),
            ('Low', '@{low}{%0.2f}'),
            ('Volume', '@{volume}'),
        ],
        formatters = {
            '@{time}' : 'datetime',
            '@{open}' : 'printf',
            '@{close}' : 'printf',
            '@{high}' : 'printf',
            '@{low}' : 'printf',
        },
        mode = 'mouse',
        show_arrow = True
    )

    main = figure(
        title = 'USD/BTC',
        x_axis_label = 'Time',
        y_axis_label = 'BTC',
        tools = TOOLS,
        height = 700,
        width = 1200,
        x_axis_type='datetime',
        x_axis_location = 'above',
        x_range = (data['time'][0], data['time'][int(len(data) * 0.15)]),
        y_range = yrange
    )

    main.add_tools(hover_tool)

    main.xaxis.major_label_orientation = pi/4
    main.grid.grid_line_alpha = 0.3

    incview = CDSView(source = source, filters = [IndexFilter([i for i in range(len(data)) if data['close'][i] > data['open'][i]])])
    decview = CDSView(source = source, filters = [IndexFilter([i for i in range(len(data)) if data['close'][i] < data['open'][i]])])

    main.segment(x0 = 'time', y0 = 'high', x1 = 'time', y1 = 'low', color="black", source = source)
    main.vbar(source = source, view = incview, x = 'time', bottom = 'open', top = 'close', width = cswidth, fill_color="green", line_color="black")
    main.vbar(source = source, view = decview, x = 'time', bottom = 'open', top = 'close', width = cswidth, fill_color="red", line_color="black")

    histogram = figure(
        title = 'Price Histogram',
        height = 700,
        width = 250,
        y_range = main.y_range,
        y_axis_location = 'right',
        x_axis_location = 'above',
    )


    counts = np.histogram(
        data['close'],
        bins = 50,
    )

    histogram.hbar(
        y = counts[1],
        right = counts[0],
        height = (yrange[1]-yrange[0])/120,
        fill_color = 'blue'
    )

    select = figure(
        title = 'Selector',
        height = 150,
        width = 1200,
        y_range = main.y_range,
        x_axis_type = 'datetime',
        tools = '',
        toolbar_location = None
    )

    range_tool = RangeTool(x_range = main.x_range)
    range_tool.overlay.fill_color = 'navy'
    range_tool.overlay.fill_alpha = .2

    select.line(x = 'time', y = 'close', source = source)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)

    p = gridplot([[main, histogram], [select]])

    if plot:
        show(p)

    else:
        return main, histogram, select, source
    
def find_missing(times: pl.Series) -> pl.LazyFrame:
    start_time = times[0]; end_time = times[-1]
    unix_minute = 1 * 60 * 1000
    return (
        pl.LazyFrame(
            data = {
                'Times': list(range(start_time, end_time, unix_minute))
                }
        ).with_columns(
            pl.col('Times').is_in(times).alias('diff')
        ).filter(
            pl.col('diff') == False
        ).with_columns( # Remove this for time in Unix ms
            pl.from_epoch(
                pl.col('Times'),
                time_unit = 'ms'
            )
        )
    )

def find_missing_days(times: pl.Series) -> pl.LazyFrame:
    start_time = times[0]; end_time = times[-1]
    unix_minute = 1 * 60 * 1000
    return (
        pl.LazyFrame(
            data = {
                'Times': list(range(start_time, end_time, unix_minute))
                }
        ).with_columns(
            pl.col('Times').is_in(times).alias('diff')
        ).filter(
            pl.col('diff') == False
        ).with_columns( # Remove this for time in Unix ms
            pl.from_epoch(
                pl.col('Times'),
                time_unit = 'ms'
            ).cast(pl.Date)
        )
    )

def getUTCUnixFromDt(year, month, day, hour, minute, second):
    # Returns ms Unix time from datetime in UTC timezone
    return time.mktime(datetime.utcfromtimestamp(time.mktime(datetime(year, month, day, hour, minute, second).timetuple())).timetuple()) * 1000

'''
AdFuller Test for stationarity of time series data
'''
def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = stats.adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

def plot_ACF(data, lags = 40):

    acf, _, pvalue = stats.acf(data, nlags = lags, qstat = True)

    pacf = stats.pacf(data, nlags = lags)

    # ACF Plot
    p1 = figure(
        title = 'ACF of BTC/USDT min data',
        x_axis_label = 'Lags',
        y_axis_label = 'Autocorrelation',
        y_range = [-1, 1],
        height = 400,
        width = 800
    )

    # ACF bars
    p1.segment(
        x0 = list(range(0, lags + 1, 1)),
        x1 = list(range(0, lags + 1, 1)),
        y0 = [0] * (lags + 1),
        y1 = acf,
        color = 'blue',
        line_width = 5
    )

    # At the 95% confidence interval z_(a-1)/sqrt(n - Li) a = 0.05, Li = Lag operator
    p1.line(
        x = list(range(0, lags + 1, 1)),
        y = [1.96 / np.sqrt(len(data) - lag) for lag in range(0, lags + 1)],
        color = 'red',
        line_dash = 'dashed'
    )

    p1.line(
        x = list(range(0, lags + 1, 1)),
        y = [-1.96 / np.sqrt(len(data) - lag) for lag in range(0, lags + 1)],
        color = 'red',
        line_dash = 'dashed'
    )

    # P-Values from ljung box test
    p1.line(
        x = list(range(0, lags + 1, 1)),
        y = pvalue,
        color = 'green',
        line_dash = 'dashed'
    )

    p1.line(
        x = list(range(0, lags + 1, 1)),
        y = [0.05] * (lags + 1),
        color = 'green',
        line_dash = 'dashed'
    )

    # PACF plot
    p2 = figure(
        title = 'PACF of BTC/USDT min data',
        x_axis_label = 'Lags',
        y_axis_label = 'Partial Autocorrelation',
        y_range = [-1, 1],
        height = 400,
        width = 800
    )

    # PACF bars
    p2.segment(
        x0 = list(range(0, lags + 1, 1)),
        x1 = list(range(0, lags + 1, 1)),
        y0 = [0] * (lags + 1),
        y1 = pacf,
        color = 'blue',
        line_width = 5
    )

    # At the 95% confidence interval z_(a-1)/sqrt(n - Li) a = 0.05, Li = Lag operator
    p2.line(
        x = list(range(0, lags + 1, 1)),
        y = [1.96 / np.sqrt(len(data) - lag) for lag in range(0, lags + 1)],
        color = 'red',
        line_dash = 'dashed'
    )

    p2.line(
        x = list(range(0, lags + 1, 1)),
        y = [-1.96 / np.sqrt(len(data) - lag) for lag in range(0, lags + 1)],
        color = 'red',
        line_dash = 'dashed'
    )

    plot = gridplot([[p1],[p2]])
    show(plot)

def plot_predictions(test, prediction, bounds = False):
    '''
    Parameters:
    test - Test data
    prediction - Model predictions
    bounds - List[[High], [Low]]
    '''
    p = figure(
        title = 'Test vs Prediction',
        x_axis_label = 'Time',
        y_axis_label = 'Close Log Returns',
        height = 600,
        width = 600
    )

    p.line(
        x = list(range(len(test))), 
        y = test,
        color = 'green'
    )

    p.line(
        x = list(range(len(test))), 
        y = prediction,
        color = 'red'
    )
    if bounds != False:
        p.line(
            x = list(range(len(bounds[0]))), 
            y = bounds[0],
            color = 'red',
            line_dash = 'dashed'
        )

        p.line(
            x = list(range(len(bounds[1]))), 
            y = bounds[1],
            color = 'red',
            line_dash = 'dashed'
        )

        p.patch(
            x = [*list(range(len(bounds[0]))), *list(range(len(bounds[1])))[::-1]],
            y = [*list(bounds[0]), *list(bounds[1])[::-1]],
            color = 'red',
            alpha = 0.2
        )

    show(p)