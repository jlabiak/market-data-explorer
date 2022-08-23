from dash import html, dcc, dash_table
import config
import dash_bootstrap_components as dbc
import data

regMenu = html.Div(
	[
	    html.Div(
	    	[
				html.Label(
					children='Please select an index for further analysis:', 
					style={'font-size': '16px'},
				),
				html.Div(
					[
						dcc.Dropdown(
							options=['S&P 500', 'Nasdaq 100', 'Russell 2000'], 
							value='Nasdaq 100', 
							id='index-selected',
							style={'font-size': '15px', 'width': '150px', 'margin-right': '3px', 'height':'30px', 'display':'inline-block'},
						),
						html.Button(
							children='Get index components', 
							id='get-tickers', 
							style={'font-size': '12px', 'height':'30px', 'display':'inline-block', 'verticalAlign':'top'},
						),
					]
				),
			]
		),
		html.Br(),
		html.Div(id='index-tickers-container'),
		html.Br(),
		html.Div(id='regression-output-freshness', hidden=True),
		dbc.Spinner(html.Div(id='regression-output-status'), spinnerClassName='spinner'),
		html.Div(id='regression-output-container'),

	]
)

tradeMenu = html.Div(
	[	
		html.B('Please select a date range and correlation measure to continue:', id='ticker-drop-label', style={'font-size': '16px'}),
		html.Div(
			[
				dcc.DatePickerRange(
			        id='trade-date-picker',
			        min_date_allowed=config.START_DATE if config.START_DATE else '2000-01-01',
                    max_date_allowed=data.get_latest_date(),
                    start_date=config.START_DATE if config.START_DATE else '2000-01-01',
                    end_date=data.get_latest_date(),
			        style={'margin-top':'10px'}
			    ),
			    dcc.RadioItems(
					options=[
						{'label': 'Pearson', 'value': 'pearson'},
						#{'label': 'Kendall', 'value': 'kendall'},
						#{'label': 'Spearman', 'value': 'spearman'},
					], 
					value='pearson', 
					id='corr-selected',
					inputStyle={'margin-right': '3px'},
					labelStyle={'margin-right': '10px','display':'block'},
					style={'margin-top':'10px','margin-bottom':'10px'},
				),
			    html.Button(
					children='Find most correlated pairs', 
					id='get-pairs', 
				),
			], 
		),
		html.Br(),
		#dbc.Spinner(html.Div(id='corr-output-container'), spinnerClassName='spinner'),
		html.Div(id='corr-output-container'),
		dcc.Interval(id='waiting'),
		html.Div(id='job-id'),
		#html.Div(id='corr-output-container-2'),
		html.Br(),
		html.Div(id='backtest-container'),
	]
)