from dash import html, dcc, Input, Output, State, callback_context as ctx, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from datetime import datetime as dt, date
import numpy as np
import time
from data import get_most_correlated, get_prices
from sklearn import linear_model

# Import config
import config

# Import app
from app import app, q, Job, conn

# Import data
import data 

@app.callback(
    Output('index-tickers-container', 'children'),
    Input('get-tickers', 'n_clicks'),
    State('index-selected', 'value'),
)
def get_index_components(n_clicks, value):
    if n_clicks:
        tickers = data.get_index_tickers(value)
        index_ticker = data.get_index_ticker(value)
        tickers.sort()
        tickers.remove(index_ticker)
        children = [
            html.Label('Please select up to 10 components of the {0} index to evaluate:'.format(value), id='ticker-drop-label', style={'font-size': '16px'}), 
            dcc.Dropdown(tickers, multi=True, id='index-tickers', style={'width':'60%', 'margin-bottom':'5px'}),
            html.B(id='analysis-state'),
            html.Br(),
            html.Div(
                [
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=config.START_DATE if config.START_DATE else '2000-01-01',
                        max_date_allowed=data.get_latest_date(),
                        start_date=config.START_DATE if config.START_DATE else '2000-01-01',
                        end_date=data.get_latest_date(),
                    )
                ], 
            ),
            html.Br(),
            html.Button('Launch {0} analysis'.format(value), id='analyse-tickers'),
        ]
        return children
    return []

@app.callback(
	Output('get-tickers', 'disabled'),
	[
		Input('get-tickers', 'n_clicks'),
		Input('index-selected', 'value'),
	],
)
def disable_button_on_click(n_clicks, _):
	context = ctx.triggered[0]['prop_id'].split('.')[0]
	if context == 'get-tickers':
		if n_clicks > 0:
			return True
		else:
			return False
	else:
		return False

@app.callback(
    [
        Output('regression-output-container', 'children'),
        Output('regression-output-status', 'children'),
    ],
    [
        Input('analyse-tickers', 'n_clicks'),
        Input('index-tickers', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
    ],
    State('index-selected', 'value'),
)
def regress_and_display(n_clicks, tickers, start_date, end_date, index_name):
    context = ctx.triggered[0]['prop_id'].split('.')[0]
    if context == 'analyse-tickers':
        # Get data for index and selected components
        index_ticker = data.get_index_ticker('{0}'.format(index_name))
        #df = data.get_prices()
        job = q.enqueue_call(func=get_prices)
        print('Queued job {}'.format(job.get_id()))
        return [[],[]]

        # Filter on tickers
        print('Filtering for tickers...')
        st = time.time()
        df = df[tickers + [index_ticker]]
        et = time.time()
        print('time to filter for tickers: {}'.format(et-st))

        # Filter on selected dates
        print('Filtering for selected dates...')
        st = time.time()
        df = df[(df.index >= dt.strptime(start_date, '%Y-%m-%d')) & (df.index <= dt.strptime(end_date, '%Y-%m-%d'))]
        et = time.time()
        print('Time to filter for selected dates: {}'.format(et-st))

        # Compute returns
        df = (df / df.shift(1)) - 1

        # Run multivariate regression
        X = sm.add_constant(df.dropna()[tickers])
        y = df.dropna()[index_ticker]
        mod = sm.OLS(y, X)
        res = mod.fit()

        # Display R-Squared
        rsquared_text = html.H4(
        	'{0} {1:.0f}% of the variance in {2} ({3}).'.format(
        		data.get_verb_for_tickers(tickers, 'explain'), 
        		res.rsquared*100, 
        		index_ticker, 
        		index_name)												
        	)

        # Describe regression results
        reg_text = html.P(
        	'Results of regression of {0} ({1}) on {2}:'.format(
        		index_ticker,
        		index_name,
        		data.get_verb_for_tickers(tickers, ''), 
        		res.rsquared*100
        	)								
        )

        # Store regression results in DataTable
        res_html = res.summary().tables[1].as_html()
        res_df = pd.read_html(res_html, header=0, index_col=0)[0]
        res_df.index.names = ['component']
        res_df = res_df[1:].reset_index()
        res_df = res_df.round(2)
        res_df['Confidence Interval'] = res_df['[0.025'].apply(lambda x: '(' + str(x)) + res_df['0.975]'].apply(lambda x: ', ' + str(x) + ')')
        res_df = res_df.drop(['[0.025', '0.975]'], axis=1)
        res_df.columns = [
            'Component',
            'Estimated Coefficient',
            'Standard Error',
            't-statistic',
            'p-value',
            'Confidence Interval',
        ]
        res_dt = dash_table.DataTable(
            data=res_df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in res_df.columns],
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['Date', 'Region']
            ],
            style_data={
                'color': 'black',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            style_header={
                'backgroundColor': 'rgb(210, 210, 210)',
                'color': 'black',
                'fontWeight': 'bold'
            },
        )

        # Plot componenet coefficient estimates
        colors = ['Positive' if c > 0 else 'Negative' for c in res.params[1:]]
        fig_reg_coeffs = px.bar(
            x=X[tickers].columns, y=res.params[1:], color=colors,
            color_discrete_sequence=['red', 'blue'],
            labels=dict(x='Component', y='Estimated Coefficient'),
        )
        fig_reg_coeffs.update_layout(showlegend=False)

        # Plot pairwise scatterplot matrix
        fig_scatter_mat = go.Figure(
            data=go.Splom(
                dimensions=[dict(label=x,values=df[x]) for x in tickers + [index_ticker]],
                diagonal=dict(visible=False),
                marker=dict(size=1),
            ),
            layout={
            	'font':{'size':10 if len(tickers) < 7 else 7},
            	'margin': go.layout.Margin(
	        		l=0,
	        		r=0,
	        		b=0,
	        		t=0,
	    		)
            },
        )
        fig_scatter_mat.update_layout({"xaxis"+str(i+1): dict(showticklabels = False) for i in range(len(tickers)+1)})
        fig_scatter_mat.update_layout({"yaxis"+str(i+1): dict(showticklabels = False) for i in range(len(tickers)+1)})

        fig_scatter = html.Div(
        	[
        		html.Br(),
		        dcc.Dropdown(
		            tickers,
		            tickers[0],
		            id='xaxis-column',
		            style={'width':'40%',},
		        ),
		        html.Br(),
                dbc.Spinner(
                    html.Div(
                        id='scatter-uni-container',
                        children = [
                            html.H4(
                                id='scatter-uni-header',
    		                ),
                            html.Br(),
                            dcc.Graph(
                            	id='indicator-graphic',
                            	config={'displayModeBar': False},
                            ),
                        ]
                    ),
                    spinnerClassName='spinner'
                )
            ]
        )

        return [[
            dbc.Tabs(
    			id='analysis-tabs',
                active_tab='multi-reg-tab',
    			children=[
    				dbc.Tab(
    					tab_id='multi-reg-tab',
    		            label='Multivariate regression',
    		            children=html.Div(
    					    [	
    					    	html.Br(),
    					        rsquared_text,
    					        reg_text,
    					        res_dt,
    					        dcc.Graph(
    					        	figure=fig_reg_coeffs, 
    					            id='reg-coeffs-fig', 
    					            config={'displayModeBar': False},
    					        ), 
    					        html.P('Scatter plot matrix:'),
    							dcc.Graph(
    								figure=fig_scatter_mat,
    							    id='scatter-plot-matrix',
    							    config={
    							    	'displayModeBar': False,
    							        'autosizable':True, 
    							    }
    							),
    						]
    					)
    		        ),
    		        dbc.Tab(
    					tab_id='uni-reg-tab',
    		            label='Univariate regression',
    		            children=fig_scatter,
    		        ),
                    dbc.Tab(
                        tab_id='opt-reg-tab',
                        label='Best subset',
                        children=html.Div(
                            [
                                html.Br(),
                                html.Label(
                                    children='Min. history to be included in analysis (securities with less will be excluded):',
                                    style={'margin-bottom': '20px'},
                                ),
                                html.Div(
                                    [
                                        html.Br(),
                                        daq.Slider(
                                            id='best-subset-slider',
                                            min=1,
                                            max=df[index_ticker].count(), 
                                            step=1, 
                                            value=df[index_ticker].count(), 
                                            handleLabel={'showCurrentValue': True, 'label': 'days'},
                                        ),
                                        # dcc.Checklist(
                                        #     id='interpolate', 
                                        #     options=[{'label':'Interpolate missing values', 'value':'interpolate'}],
                                        #     inputStyle={'margin-right': '3px'},
                                        # ),
                                        html.Br(),
                                        html.Button(
                                            children='Find best subset', 
                                            id='find-best-subset',
                                            style={'font-size': '12px', 'height':'30px', 'margin-top':'10px'},
                                        ),
                                        html.Div(id='best-subset-out', style={'margin-top':'25px'})
                                    ]
                                )
                            ]
                        )
                    ),
                ]
            )
        ],[]]
    return [[],[]]

@app.callback(
    [
        Output('analyse-tickers', 'disabled'),
        Output('analysis-state', 'children'),
    ],
    [
        Input('index-tickers', 'value'),
        Input('analyse-tickers', 'n_clicks'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
    ],
)
def disable_analysis(values, n_clicks, start_date, end_date):
    context = ctx.triggered[0]['prop_id'].split('.')[0]
    if context == 'index-tickers':
        if len(values) < 11:
            children = ''
            return [False, children]
        else:
            children = 'More than 10 components selected. Please remove components to proceed.'
            return [True, children]
    elif context == 'date-picker-range':
        children = ''
        return [False, children]
    else:
        children = ''
        return [True, children]

@app.callback(
    [
        Output('indicator-graphic', 'figure'),
        Output('scatter-uni-header','children'),
    ],
    [
        Input('xaxis-column', 'value'),
        Input('analyse-tickers', 'n_clicks'),
    ],
    [
        State('index-tickers', 'value'),
        State('index-selected', 'value'),
        State('date-picker-range', 'start_date'),
        State('date-picker-range', 'end_date'),
    ],
)  
def update_graph(xaxis_column_name, n_clicks, tickers,
                 index_name, start_date, end_date):      
    if n_clicks is None:
        PreventUpdate

    # Get data for index and selected components
    index_ticker = data.get_index_ticker('{0}'.format(index_name))
    df = data.get_prices()

    # Filter on tickers
    print('Filtering for tickers...')
    st = time.time()
    df = df[tickers + [index_ticker]]
    et = time.time()
    print('time to filter for tickers: {}'.format(et-st))

    # Filter on selected dates
    st = time.time()
    df = df[(df.index >= dt.strptime(start_date, '%Y-%m-%d')) & (df.index <= dt.strptime(end_date, '%Y-%m-%d'))]
    et = time.time()
    print('time to filter df: {}'.format(et-st))

    # Compute returns
    df = (df / df.shift(1)) - 1

    fig_uni_scatter = px.scatter(
        x=df[xaxis_column_name],
        y=df[index_ticker],
        trendline='ols',
        hover_name=df.index,
    )

    fig_uni_scatter.update_layout(margin={'l': 0, 'b': 0, 't': 0, 'r': 0}, hovermode='closest')
    fig_uni_scatter.update_xaxes(title=xaxis_column_name)
    fig_uni_scatter.update_yaxes(title=index_ticker)

    results = px.get_trendline_results(fig_uni_scatter)

    rsquared_uni_str = 'Regression eq.: Y = {0:.2f} + {1:.2f}X'.format(
        results.iloc[0]["px_fit_results"].params[0],
        results.iloc[0]["px_fit_results"].params[1])

    rsquared_uni_text = '{0} on its own explains {1:.0f}% of the variance in {2} ({3}).'.format(
        xaxis_column_name,
        results.iloc[0]["px_fit_results"].rsquared*100, 
        index_ticker, 
        index_name)                                             

    fig_uni_scatter.update_layout(
        title={
            'text': rsquared_uni_str,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
    )

    return [fig_uni_scatter, rsquared_uni_text]

@app.callback(
    Output('best-subset-out', 'children'),
    [
        Input('find-best-subset', 'n_clicks'),
        Input('best-subset-slider', 'value'),
    ],
    [
        State('index-selected', 'value'),
        State('date-picker-range', 'start_date'),
        State('date-picker-range', 'end_date'),
    ]
)
def find_best_subset(n_clicks, num_days, index_name, start_date, end_date):
    if n_clicks is None: 
        PreventUpdate
    else:
        context = ctx.triggered[0]['prop_id'].split('.')[0]
        if context == 'best-subset-slider':
            return []
        else:
            # Get data for index and index components
            tickers = data.get_index_tickers(index_name)
            index_ticker = data.get_index_ticker('{0}'.format(index_name))
            df = data.get_prices()

            # Filter on tickers
            print('Filtering for tickers...')
            st = time.time()
            df = df[tickers]
            et = time.time()
            print('Time to filter for tickers: {}'.format(et-st))

            # Filter on selected dates
            print('Filtering for selected dates...')
            st = time.time()
            df = df[(df.index >= dt.strptime(start_date, '%Y-%m-%d')) & (df.index <= dt.strptime(end_date, '%Y-%m-%d'))]
            et = time.time()
            print('Time to filter for selected dates: {}'.format(et-st))

            # Compute returns
            df = (df / df.shift(1)) - 1

            # Filter for securities with desired history
            df = df.loc[:, df.count() >= num_days]

            # Run LARS to find best subset
            X_cols = [t for t in tickers if t in df.columns and t != index_ticker]
            X = df.dropna()[X_cols].values
            y = df.dropna()[index_ticker].values
            mod = linear_model.Lars(n_nonzero_coefs=10, normalize=False)
            res = mod.fit(X, y)
            best_subset = [X_cols[i] for i in res.active_]
            best_subset_coef = [round(res.coef_[i], 4) for i in res.active_]

            # Store regression results in DataTable
            res_df = pd.DataFrame({'Component':best_subset, 'Estimated Coefficient':best_subset_coef})
            res_dt = dash_table.DataTable(
                data=res_df.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_data={ 'border': '1px solid black'},
                style_header={ 'border': '1px solid black'},                    
            )

            return [
                html.P(
                    'Of the {0} (out of {1}) {2} components with at least {3} days of price history the following are the 10 that best explain {4} returns:'.format(
                        len(X_cols),
                        len(tickers)-1,
                        index_name,
                        num_days,
                        index_ticker)                                             
                ),
                res_dt
            ]

@app.callback(
    Output('find-best-subset', 'disabled'),
    [
        Input('find-best-subset', 'n_clicks'),
        Input('best-subset-slider', 'value'),
    ],
)
def disable_button_on_click(n_clicks, _):
    context = ctx.triggered[0]['prop_id'].split('.')[0]
    if context == 'find-best-subset':
        if n_clicks > 0:
            return True
        else:
            return False
    else:
        return False

@app.callback(
    [
        Output('corr-output-container', 'children'),
        Output('corr-output-container', 'hidden'),
        # Output('job-id', 'value'),
    ],
    [
        Input('get-pairs', 'n_clicks'),
        # Input('waiting', 'n_intervals'),
        Input('trade-date-picker', 'start_date'),
        Input('trade-date-picker', 'end_date'),
        Input('corr-selected', 'value'),
    ],
    # State('job-id', 'value'),
)
# def find_pairs(n_clicks, n_intervals, start_date, end_date, corr_meth, job_id):
def find_pairs(n_clicks, start_date, end_date, corr_meth):
    if n_clicks:
        context = ctx.triggered[0]['prop_id'].split('.')[0]
        if context == 'get-pairs':
            pairs = data.get_most_correlated(start_date, end_date, corr_meth)
            pairs['corr'] = pairs['corr'].apply(lambda x: f'{x:.4f}')
            # job = q.enqueue_call(func=get_most_correlated, args=(start_date, end_date, corr_meth,))
            # print('Queued job {}'.format(job.get_id()))
            # return [dbc.Spinner(spinnerClassName='spinner', fullscreen=True), False, job.get_id()]
            return [
                [
                    html.B(
                        id='pairs-table-title',
                        children='Select pairs (rows) from the table below to include in the strategy backtest:',
                    ),
                    html.Div([
                        dash_table.DataTable(
                            id='pairs-table',
                            columns=[
                                {'name': 'Ticker 1', 'id': 'ticker1'},
                                {'name': 'Ticker 2', 'id': 'ticker2'},
                                {'name': 'Corr. Coeff.', 'id': 'corr'},
                            ],
                            data=pairs[['ticker1','ticker2','corr']].to_dict('records'),
                            style_cell={'textAlign': 'center'},
                            style_data={ 'border': '1px solid black'},
                            style_header={ 'border': '1px solid black'},                    
                            row_selectable='multi',
                            row_deletable=True,
                            selected_rows=[],
                            page_current= 0,
                            page_size= 10,
                        )],
                        style={'margin-top':'20px'}
                    )
                ], 
                False,
            ]
    return [[], True]
    #     elif context == 'waiting':
    #         job = Job.fetch(job_id, connection=conn)
    #         if job.is_finished:
    #             pairs = job.result
    #             pairs['corr'] = pairs['corr'].apply(lambda x: f'{x:.4f}')
    #             return [[
    #                 html.B(
    #                     id='pairs-table-title',
    #                     children='Select pairs (rows) from the table below to include in the strategy backtest:',
    #                 ),
    #                 html.Div([
    #                     dash_table.DataTable(
    #                         id='pairs-table',
    #                         columns=[
    #                             {'name': 'Ticker 1', 'id': 'ticker1'},
    #                             {'name': 'Ticker 2', 'id': 'ticker2'},
    #                             {'name': 'Corr. Coeff.', 'id': 'corr'},
    #                         ],
    #                         data=pairs[['ticker1','ticker2','corr']].to_dict('records'),
    #                         style_cell={'textAlign': 'center'},
    #                         style_data={ 'border': '1px solid black'},
    #                         style_header={ 'border': '1px solid black'},                    
    #                         row_selectable='multi',
    #                         row_deletable=True,
    #                         selected_rows=[],
    #                         page_current= 0,
    #                         page_size= 10,
    #                     )],
    #                     style={'margin-top':'20px'}
    #                 ),
    #             ], False, '']
    #         else:
    #             return [dbc.Spinner(spinnerClassName='spinner', fullscreen=True), False, job.get_id()]
    # return [[],False, '']

# @app.callback(
#     Output('waiting', 'disabled'),
#     Input('job-id', 'value'),
# )
# def disable_interval(job_id):
#     if job_id == '':
#         return True
#     return False

@app.callback(
    Output('get-pairs', 'disabled'),
    [
        Input('get-pairs', 'n_clicks'),
        Input('trade-date-picker', 'start_date'),
        Input('trade-date-picker', 'end_date'),
        Input('corr-selected', 'value'),
    ],
)
def disable_button_on_click(n_clicks, start, end, corr_meth):
    context = ctx.triggered[0]['prop_id'].split('.')[0]
    if context == 'get-pairs':
        if n_clicks > 0:
            return True
        else:
            return False
    else:
        return False

@app.callback(
    Output('backtest-container', 'children'),
    [
        Input('pairs-table', 'selected_rows'),
        Input('trade-date-picker', 'start_date'),
        Input('trade-date-picker', 'end_date'),
        Input('corr-selected', 'value'),
    ],
)
def get_backtest_params(pairs, start, end, corr_meth):
    context = ctx.triggered[0]['prop_id'].split('.')[0]
    if context != 'pairs-table' or len(pairs) == 0:
        return []
    else:
        return html.Div(
            [   
                html.Label(
                    children='Select threshold for entering trades (relative performance over trailing 5d vs. trailing 63d):',
                    style={
                        'font-weight':'bold',
                        'margin-bottom':'25px',
                    },
                ),
                dcc.Slider(0, 4, 0.5,
                   value=2,
                   id='thres-slider',
                ),
                html.Label(
                    children='Select trade duration type and threshold:',
                    style={
                        'font-weight':'bold',
                        'margin-top':'25px',
                        'margin-bottom':'25px',
                    },
                ),
                dcc.RadioItems(
                    options=[
                        {'label': 'Fixed (days)', 'value': 'fixed'},
                        {'label': 'Convergence (relative performance over trailing 5d vs. trailing 63d)', 'value': 'convergence'},
                    ], 
                    value='fixed', 
                    inputStyle={'margin-right': '3px'},
                    labelStyle={'margin-right': '10px','display':'block'},
                    id='duration-type',
                ),
                html.Br(),
                dcc.Slider(
                    min=1,
                    max=63,
                    step=None,
                    marks={
                        1:'1',
                        2:'2',
                        5:'5',
                        10:'10',
                        21:'21',
                        42:'42',
                        63:'63',
                    },
                    value=5,
                    id='duration-slider'
                ),
                html.Div(
                    children=[
                        html.Label(
                            children='Enter position size (in USD):',
                            style={
                                'font-weight':'bold',
                                'margin-top':'25px',
                                'margin-bottom':'25px',
                            },
                        ),   
                        dcc.Input(
                            id='tradesize-input',
                            type='text',
                            value=1e6,
                            placeholder='Trade size',
                            style={'width': '100px', 'margin-left': '10px'}
                        ),
                    ],
                    style={'display':'inline-block'},
                ),             
                html.Br(),
                html.Button(
                    id='compute-pnl-button',
                    children='Run backtest for selected pairs',
                    disabled=True,
                    style={'margin-bottom':'10px'},
                ),
                dbc.Spinner(html.Div(id='backtest-output'), spinnerClassName='spinner'),
            ],
        )  

@app.callback(
    Output('compute-pnl-button', 'disabled'),
    [
        Input('pairs-table', 'selected_rows'),
        Input('compute-pnl-button', 'n_clicks'),
    ],
)
def enable_compute_pnl(pairs, n_clicks):
    context = ctx.triggered[0]['prop_id'].split('.')[0]
    if n_clicks is None:
        PreventUpdate
    if context == 'pairs-table':
        if len(pairs) > 0:
            return False
        else:
            return True
    elif context == 'compute-pnl-button':
        if (n_clicks) & (n_clicks > 0):
            return True
        else:
            return False

@app.callback(
    [
        Output('duration-slider', 'min'),
        Output('duration-slider', 'max'),
        Output('duration-slider', 'step'),
        Output('duration-slider', 'marks'),
        Output('duration-slider', 'value'),
    ],
    Input('duration-type', 'value'),
)
def get_duration_slider(type):
    if type == 'fixed':
        return [
            1, 
            63,
            None,
            {
                1:'1',
                2:'2',
                5:'5',
                10:'10',
                21:'21',
                42:'42',
                63:'63',
            },
            5,
        ]
    elif type == 'convergence':
        return [
            0, 
            4, 
            0.5,
            {},
            0.5,
        ]

@app.callback(
    Output('backtest-output', 'children'),
    [
        Input('compute-pnl-button', 'n_clicks'),
        Input('tradesize-input', 'value'),
        Input('thres-slider', 'value'),
        Input('duration-type', 'value'),
        Input('duration-slider', 'value'),
    ],
    [
        State('pairs-table', 'selected_rows'),
        State('pairs-table', 'data'),
        State('trade-date-picker', 'start_date'),
        State('trade-date-picker', 'end_date'),
    ],
)
def run_backtest(n_clicks, trade_size, entry_thres, exit_type, exit_thres, selected, pairs, start_date, end_date):
    if n_clicks:
        # Get prices data
        st = time.time()
        prices = data.get_prices()
        et = time.time()
        print('Took {} seconds to load prices data for backtest.'.format(et - st))

        # Filter on selected dates
        st = time.time()
        prices = prices[(prices.index >= dt.strptime(start_date, '%Y-%m-%d')) & (prices.index <= dt.strptime(end_date, '%Y-%m-%d'))]
        et = time.time()
        print('time to filter df: {}'.format(et-st))

        total_pnl = pd.Series(name='pnl')
        for i in selected:
            pnl = data.get_daily_pnl(prices, pairs[i]['ticker1'], pairs[i]['ticker2'], trade_size, entry_thres, exit_type, exit_thres, start_date, end_date)
            #pnl.to_csv('../data/pnl_' + '_' + pairs[i]['ticker1'] + '_' + pairs[i]['ticker2'] + '.csv')
            total_pnl = total_pnl.add(pnl, fill_value=0)
        
        total_pnl = total_pnl.fillna(0)

        fig_cum_pnl = px.line(
            total_pnl.cumsum().reset_index(),
            x='date',
            y='pnl',
        )

        fig_cum_pnl.update_layout(margin={'l': 0, 'b': 0, 't': 0, 'r': 0}, hovermode='closest')

        # Compute backtest stats
        pnl_annual = total_pnl.mean()*252
        sd_annual = total_pnl.std()*(252**(1/2))
        sharpe = pnl_annual / sd_annual
        max_draw = data.compute_max_draw(total_pnl.cumsum().tolist())

        res_tab = dash_table.DataTable(
            id='backtest-table',
            columns=[
                {'name': 'Expected Annual PnL ($)', 'id': 'pnl'},
                {'name': 'Std. Dev. ($)', 'id': 'sd'},
                {'name': 'Sharpe Ratio', 'id': 'sharpe'},
                {'name': 'Max. Drawdown ($)', 'id': 'maxdraw'},
            ],
            data=[{
                'pnl': round(pnl_annual, 2),
                'sd': round(sd_annual, 2),
                'sharpe': round(sharpe, 2),
                'maxdraw': round(max_draw, 2),
            }],
            style_cell={'textAlign': 'center'},
            style_data={ 'border': '1px solid black'},
            style_header={ 'border': '1px solid black'},                    
        )

        return html.Div([
            dcc.Graph(
                id='cum-pnl',
                figure=fig_cum_pnl,
                config={
                    'displayModeBar': False,
                },
            ),
            html.Br(),
            res_tab,
        ])
    return []