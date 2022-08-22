import os
import psycopg2
import subprocess

# Set to 'local' or 'heroku'
RUN_ENV = 'local'

START_DATE = '2000-01-01'
PATH_TO_INDEX_COMPONENTS = 'data/index_components.csv'
#DB_URL = 'sqlite:///./data/equity_data.db' #'postgresql://localhost/johnlabiak'

if RUN_ENV == 'local':
	DB_URL = os.environ['DATABASE_URL']
elif RUN_ENV == 'heroku':
	HEROKU_APP_NAME = 'market-data-explorer'
	DB_URL = subprocess.run(
	    ['heroku', 'config:get', 'DATABASE_URL', '--app', HEROKU_APP_NAME],
	    capture_output=True
	).stdout 