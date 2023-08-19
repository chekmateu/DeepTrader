import polars as pl
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import RangeTool, HoverTool, ColumnDataSource, CDSView, IndexFilter
from math import pi

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