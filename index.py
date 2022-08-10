# Import dash html, core components, IO, bootstrap components
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Import layouts and callbacks
from layouts import regMenu, tradeMenu
import callbacks

# Import app
from app import app
# Import server for deployment
from app import server

# Define sidebar style
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'padding': '2rem 1rem',
    'background-color': '#f8f9fa',
}

# Define main content style
CONTENT_STYLE = {
    'margin-left': '18rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
}

sidebar = html.Div(
    [
        html.H2('Market Data Explorer', className='display-5'),
        html.Hr(),
        dbc.Nav(
            [dbc.NavLink('Home', href='/', active='exact')],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H2('Analyses', className='lead'),
        dbc.Nav(
            [
                dbc.NavLink('Index Regression', href='/regression', active='exact'),
                dbc.NavLink('Pair Trading', href='/trading', active='exact'),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id='page-content', style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id='url'), sidebar, content])

@app.callback(
    Output('page-content', 'children'), 
    Input('url', 'pathname'),
)
def render_page_content(pathname):
    if pathname == '/':
        return html.Div([dcc.Markdown('''
            ### Introduction
            This application is a project built by [John Labiak](https://jlabiak.github.io) using Plotly's Dash,
            faculty.ai's Dash Bootstrap Components, Pandas, statsmodels for linear regression modeling, and custom functions. 
            Using historical Yahoo Finance data, this application allows the user to see how much of the variance of a selected index 
            can be explained by a basket of component securities and allows the user to evaluate various equity pair-trading strategies.
        ''')],className='home')
    elif pathname == '/regression':
        return regMenu
    elif pathname == '/trading':
        return tradeMenu
    else:
       # If the user tries to reach a different page, return a 404 message
       return dbc.Jumbotron(
           [
               html.H1('404: Not found', className='text-danger'),
               html.Hr(),
               html.P(f'The pathname {pathname} was not recognized...'),
           ]
       )

# Call app server
if __name__ == '__main__':
    # Set debug to false when deploying app
    app.run_server(debug=True)