from dash import Dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

app = Dash(__name__, 
           external_stylesheets=[dbc.themes.SANDSTONE],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},],)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'redis',
#     'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
# })
TIMEOUT = 60*60*24

# Set app server to variable for deployment
server = app.server

# Set app callback exceptions to true
app.config.suppress_callback_exceptions = True

# Set app title
app.title = 'Market Data Explorer'