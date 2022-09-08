from apscheduler.schedulers.blocking import BlockingScheduler
import redis
from data import get_prices_from_db
import pickle
import zlib
import os

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=1)
def scheduled_job():
    print('Running scheduled job...')
    df = get_prices_from_db()
    print('Connecting to Redis server at {}'.format(os.environ.get('REDIS_URL', '')))
    r = redis.from_url(os.environ.get('REDIS_URL', ''))
    print(r.ping())
    #r = redis.from_url('redis://localhost:6379')
    #r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('prices', zlib.compress(pickle.dumps(df)))

sched.start()
