from apscheduler.schedulers.blocking import BlockingScheduler
import update_db
import redis
from data import get_prices_from_db
from app import r
import pickle
import zlib
import sys
import config

sched = BlockingScheduler()

@sched.scheduled_job('cron', day_of_week='*', hour=1)
def fetch_new_data():
    update_db()
    return True

@sched.scheduled_job('cron', day_of_week='*', hour=3)
def cache_data():
    print('Running scheduled job...')
    df = get_prices_from_db()
    df = df[df.index > config.START_DATE]
    compressed_df = zlib.compress(pickle.dumps(df))
    print('Caching df of size {}'.format(sys.getsizeof(compressed_df) / 1e6))
    r.set('prices', compressed_df)
    return True

sched.start()
