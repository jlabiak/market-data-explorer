from dash import Dash
import dash_bootstrap_components as dbc
from flask_caching import Cache
import os
import redis
from urllib.parse import urlparse

# from rq import Queue
# from rq.job import Job
# from worker import conn

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

# q = Queue(connection=conn)
# redis_url = urlparse(os.getenv('REDIS_URL', 'redis://localhost:6379'))
# r = redis.Redis(host=redis_url.hostname, port=redis_url.port, username=redis_url.username, password=redis_url.password, ssl=True, ssl_cert_reqs=None)
r = redis.from_url('redis://localhost:6379')

# Set app server to variable for deployment
server = app.server

# Set app callback exceptions to true
app.config.suppress_callback_exceptions = True

# Set app title
app.title = 'Market Data Explorer'
