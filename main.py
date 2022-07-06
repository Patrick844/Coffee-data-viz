# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go


app = dash.Dash(__name__)
server = app.server



app.layout = html.Div(children=[
    html.H2("Buy an Espresso machine or Coffee at a shop ?"),
    dcc.Graph(
        id='Should I buy an Espresso Machine or Coffee Vending Machine?',
    ),
    html.H6("The sale number of Espresso machine"),
    dcc.Slider(
        0,
        100,
        step=1,
        value=2,
        id='Frequency1'
    ),
    
    html.H6("The sale number of Coffee ending machine"),
    dcc.Slider(
        0,
        100,
        step=1,
        value=2,
        id='Frequency2'
    ),
    html.H6("The price of your Epresso Machine"),
    dcc.Slider(
        0,
        2000,
        step=1,
        value=1500,
        id='Price1'
    ),
    html.H6("The price of your Coffee Vending Machine"),
    dcc.Slider(
        0,
        2000,
        step=1,
        value=300,
        id='Price2'
    )
])


@app.callback(
    Output('Should I buy an Espresso Machine or Coffee Vending Machine?', 'figure'),
    Input('Frequency1', 'value'),
    Input('Frequency2', 'value'),
    Input('Price1', 'value'),
    Input('Price2', 'value'))
def update_cpp(Frequency1, Frequency2, Price1, Price2):
    time = 365
    x = np.linspace(1, time, time)
    x1 = x * Frequency1
    x2 = x * Frequency2
    y1 = -Price1 + Frequency1 * x1 * 0.2
    y2 = -Price2 + Frequency2 * x2 * 0.1
    y3 = [0 for _ in range(time)]
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=x, y=y1, name='Profit on Epresso',
                           line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y2, name='Profit on Vending',
                           line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y3, name='Return on investment',
                           line=dict(color='green', width=4)))
# Edit the layout
    fig.update_layout(title='Should you buy an Espresso Machine? Check this graph!',
                    xaxis_title='Days',
                    yaxis_title='Euro')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
