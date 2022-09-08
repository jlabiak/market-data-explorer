from apscheduler.schedulers.blocking import BlockingScheduler
import redis
from data import get_prices_from_db
from app import r
import pickle
import zlib
import os
import sys

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=3)
def scheduled_job():
    print('Running scheduled job...')
    df = get_prices_from_db()
    df = df[df.index > '2011-01-01']
    compressed_df = zlib.compress(pickle.dumps(df))
    print('Setting df of size {}'.format(sys.getsizeof(compressed_df) / 1e6))
    r.set('prices', compressed_df)

sched.start()
