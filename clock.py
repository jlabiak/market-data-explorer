from apscheduler.schedulers.blocking import BlockingScheduler
from data import get_prices

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=1)
def scheduled_job():
    _ = get_prices()

sched.start()
