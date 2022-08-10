import os
import psycopg2

START_DATE = '2000-01-01'
END_DATE = '2022-07-01'
PATH_TO_INDEX_COMPONENTS = '../data/index_components.csv'
DB_URL = os.environ['DATABASE_URL'] # set to 'postgresql://johnlabiak:@localhost/johnlabiak'