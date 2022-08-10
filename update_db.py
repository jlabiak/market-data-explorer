import pandas as pd
import config
from data import create_connection
import populate_db

def main():
    print('Querying SQL for latest date...')
    database = config.DB_URL
    conn = create_connection(database)
    result = conn.execute('SELECT MAX(date) FROM prices')
    sql_result = result.fetchall()
    latest_date = pd.to_datetime(sql_result[0][0]).strftime('%Y-%m-%d')
    print('Latest date is {}'.format(latest_date))

    print('Getting index components...')
    tickers = populate_db.get_tickers()

    print('Getting daily price data from {}...'.format(latest_date))
    data = populate_db.get_yahoo_data(tickers['ticker'].tolist(), start=latest_date)
    data = data.reset_index().merge(tickers, on='ticker')
    data = data.set_index('date')

    print('Updating daily price data in {}...'.format(config.DB_URL))
    _ = populate_db.write_to_db(data, config.DB_URL, 'append')

if __name__ == '__main__':
    main()