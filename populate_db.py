import pandas as pd
import yfinance as yf
import config
from data import create_connection
import get_index_components

# Get components of S&P 500, Nasdaq 100, and Russell 2000
def get_tickers():
    if config.PATH_TO_INDEX_COMPONENTS:
        try:
            tickers = pd.read_csv(config.PATH_TO_INDEX_COMPONENTS, index_col=0)
            return tickers
        except Exception as e:
            print('Failed parsing tickers from {}: {}'.format(config.PATH_TO_INDEX_COMPONENTS, e))
    return get_index_components()

# Get data from Yahoo Finance
def get_yahoo_data(tickers, which_price='Adj Close', **kwargs):
    data = None
    try:
        data = yf.download(tickers, threads=False, **kwargs)
        data = data[which_price]
        data = pd.melt(data, var_name='ticker', value_name='price', ignore_index=False)
        data = data.sort_index()
        data.index.names = ['date']
    except Exception as e:
        print('Failed retrieving data from Yahoo Finance:\n{}'.format(e))
    return data

# Write data to SQL DB
def write_to_db(data, db_url, if_exists):
    conn = create_connection(db_url)

    if conn:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS prices (
                    date DATE,
                    ticker VARCHAR(20),
                    price REAL,
                    indices VARCHAR(50)
                );
            """
        )
        try:
            print('Writing data to {}...'.format(db_url))
            data.to_sql('prices', con=conn, chunksize=1000000, if_exists=if_exists)
            print('Wrote {} records to {}.'.format(len(data), db_url))
        except Exception as e:
            print('Failed writing data to {}:{}'.format(db_url, e))
        conn.close()

def main():
    print('Getting index components...')
    tickers = get_tickers()

    print('Getting daily price data from {}...'.format(config.START_DATE if config.START_DATE else '2000-01-01'))
    data = get_yahoo_data(tickers['ticker'].tolist(), start=config.START_DATE if config.START_DATE else '2000-01-01')
    data = data.reset_index().merge(tickers, on='ticker')
    data = data.set_index('date')

    print('Writing daily price data to {}...'.format(config.DB_URL))
    _ = write_to_db(data, config.DB_URL, 'replace')

if __name__ == '__main__':
    main()
