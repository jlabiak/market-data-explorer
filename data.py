from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import config
from datetime import datetime as dt
import time
import sys
import tempfile

from app import cache, TIMEOUT

def read_sql_tmpfile(query, db_url, **kwargs):
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
           query=query, head="HEADER"
        )
        engine = create_engine(db_url)
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        conn.close()
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile, **kwargs)
        return df

@cache.memoize(timeout=TIMEOUT)
def get_prices():
    print('Loading price data...')
    st = time.time()
    # chunks = pd.read_sql_table('prices', config.DB_URL, chunksize=100000)
    # dfs = []
    # for c in chunks:
    #     dfs.append(c)
    #     print('chunk size {}MB'.format(sys.getsizeof(c) / 1e6))
    # df = pd.concat(dfs)
    # df['date'] = pd.to_datetime(df['date'])
    df = read_sql_tmpfile(
        query='SELECT date,ticker,price FROM prices', 
        db_url=config.DB_URL, 
        index_col='date', 
        parse_dates=True, 
        dtype={'ticker': 'str', 'price': 'float'}, 
        engine='pyarrow'
    )
    print('Returned df of size {}MB'.format(sys.getsizeof(df) / 1e6))
    et = time.time()
    print('Time to load price data: {}'.format(et-st))

    # Pivot dataframe
    print('Pivoting df...')
    st = time.time()
    df = df.pivot(columns='ticker', values='price')
    et = time.time()
    print('Time to pivot df: {}'.format(et-st))

    return df

# @cache.memoize(timeout=TIMEOUT)
# def get_prices():
#     print('Loading price data...')
#     st = time.time()

#     print('Querying SQL...')
#     database = config.DB_URL
#     # Create a database connection
#     conn = create_connection(database)
#     result = conn.execute('SELECT date,ticker,price FROM prices')
#     print('Returned SQL result of size {}MB'.format(sys.getsizeof(result) / 1e6))
#     sql_results=result.fetchall()
#     print('Fetched SQL result of size {}MB'.format(sys.getsizeof(sql_results) / 1e6))
#     conn.close()

#     print('Converting to df...')
#     df=pd.DataFrame(sql_results, columns=['date','ticker','price'])
#     df['date'] = pd.to_datetime(df['date'])
#     print('Returned df of size {}MB'.format(sys.getsizeof(df) / 1e6))

#     et = time.time()
#     print('Time to load price data: {}'.format(et-st))

#     # Pivot dataframe
#     print('Pivoting df...')
#     st = time.time()
#     df = df.pivot(index='date', columns='ticker', values='price')
#     et = time.time()
#     print('Time to pivot df: {}'.format(et-st))

#     return df

@cache.memoize(timeout=TIMEOUT)
def get_latest_date():
    database = config.DB_URL
    conn = create_connection(database)
    result = conn.execute('SELECT MAX(date) FROM prices')
    sql_result = result.fetchall()
    return pd.to_datetime(sql_result[0][0]).strftime('%Y-%m-%d')

def create_connection(db_url):
    """ Create a database connection to the SQL database
        specified by the db_url
    :param db_url: database file
    :return: Connection object or None
    """
    conn = None
    try:
        engine = create_engine(db_url)
        conn = engine.connect()
        #conn = db.connect(db_file, check_same_thread=False)
        print('Connection to database successful.')
    except Exception as e:
        print('Connection to database failed.')
        raise Exception

    return conn

def get_index_tickers(index_name):
    tickers = pd.read_csv(config.PATH_TO_INDEX_COMPONENTS, index_col=0)
    tickers = tickers[tickers['indices'].str.contains(index_name)]['ticker'].unique().tolist()
    return tickers

def get_index_tickers_dates(index):
    database = config.DB_URL
    # Create a database connection
    conn = create_connection(database)
    result = conn.execute('SELECT * FROM prices WHERE indices LIKE "%{0}%"'.format(index))
    sql_results=result.fetchall()
    conn.close()

    df=pd.DataFrame(sql_results,columns=['date','ticker','price','prev_1d_ret','indices'])
    tickers = df['ticker'].unique().tolist()
    tickers.sort()
    min_date = pd.to_datetime(df['date'].min()).date()
    max_date = pd.to_datetime(df['date'].max()).date()
    return [tickers, min_date, max_date]

def get_index_ticker_data(index, tickers):
    database = config.DB_URL
    # Create a database connection
    conn = create_connection(database)
    index_ticker = get_index_ticker('{0}'.format(index))
    result = conn.execute('SELECT * FROM prices WHERE indices LIKE "%{0}%" AND ticker IN ({1})'.format(index, ','.join(['"' + t + '"' for t in tickers + [index_ticker]])))
    sql_results=result.fetchall()
    conn.close()
    df=pd.DataFrame(sql_results,columns=['date','ticker','price','prev_1d_ret','indices'])
    return df

def get_index_ticker(index):
    index_to_ticker = {
        'S&P 500': '^GSPC',
        'Nasdaq 100': 'NDX',
        'Russell 2000': '^RUT',
    }
    return index_to_ticker[index]

def get_verb_for_tickers(tickers, verb):
        if len(tickers) == 0:
            return ' ' + verb
        elif len(tickers) == 1:
            return str(tickers[0]) + ' ' + verb + 's'
        elif len(tickers) == 2:
            return str(tickers[0]) + ' and ' + str(tickers[1]) + ' ' + verb
        else:
            return ', '.join(tickers[:-1]) + ', and ' + str(tickers[-1]) + ' ' + verb

def get_most_correlated(start_date, end_date, corr_meth, n=50):
    # Get data
    st = time.time()
    df = get_prices()
    et = time.time()
    print('Took {} seconds to load df.'.format(et - st))

    # Filter on selected dates
    st = time.time()
    df = df[(df.index >= dt.strptime(start_date, '%Y-%m-%d')) & (df.index <= dt.strptime(end_date, '%Y-%m-%d'))]
    et = time.time()
    print('time to filter df: {}'.format(et-st))

    # Compute returns
    df = (df / df.shift(1)) - 1
    
    # Compute pairwise correlations
    print('Computing correlations...')
    st = time.time()
    corrm = df.corr(method=corr_meth)
    #corrm = pd.DataFrame(np.corrcoef(df.values, rowvar=False), columns=df.columns)
    et = time.time()
    print('Took {} seconds to compute correlations.'.format(et - st))
    print('Size of corrm: {}MB'.format(sys.getsizeof(corrm) / 1e6))

    # Find n largest pairwise correlations
    # sol = (corrm.where(np.triu(np.ones(corrm.shape), k=1).astype(bool))
    #               .stack()
    #               .sort_values(ascending=False))
    # print('Size of sol: {}MB'.format(sys.getsizeof(sol) / 1e6))
    mask = np.ones(corrm.shape, dtype='bool')
    print('Size of mask: {}MB'.format(sys.getsizeof(mask) / 1e6))
    mask[np.triu_indices(len(corrm))] = False
    print('Size of new mask: {}MB'.format(sys.getsizeof(mask) / 1e6))
    corrm = corrm.mask(~mask, 0)
    print('Size of masked corr: {}MB'.format(sys.getsizeof(corrm) / 1e6))
    pairs = corrm.stack().nlargest(n).index
    print('Size of pairs: {}MB'.format(sys.getsizeof(pairs) / 1e6))

    # Format DataFrame with results
    print('Formatting results...')
    pairs = pd.DataFrame(pairs, columns=['pair'])
    pairs['corr'] = pairs['pair'].apply(lambda x: round(corrm.loc[x],4))
    pairs['ticker1'] = pairs['pair'].apply(lambda x: x[0])
    pairs['ticker2'] = pairs['pair'].apply(lambda x: x[1])

    return pairs

def get_daily_pnl(prices, ticker1, ticker2, trade_size, entry_thres, exit_type, exit_thres, start_date, end_date):
    trade_size = int(trade_size)
    positions_t1 = []
    positions_t2 = []
    pos_t1 = 0
    pos_t2 = 0

    # Relevant for 'duration' exit_type
    holding_period = 0

    with pd.option_context('mode.chained_assignment', None):
        prices = prices[[ticker1, ticker2]]
        prices['ratio'] = prices[ticker1] / prices[ticker2]
        prices['ratio_mean_63d'] = prices['ratio'].shift(1).rolling(63, min_periods=10).mean()
        prices['ratio_mean_3d'] = prices['ratio'].rolling(3, min_periods=2).mean()
        prices['ratio_std_63d'] = prices['ratio'].shift(1).rolling(63, min_periods=10).std()
        prices['ratio_std_63d'] = prices['ratio_std_63d'].apply(lambda x: max(x, 0.01))
        prices['ratio_z'] = (prices['ratio_mean_3d'] - prices['ratio_mean_63d']) / prices['ratio_std_63d']

        for t1, t2, z in np.array(prices[[ticker1, ticker2, 'ratio_z']]):
            if pos_t1 == 0: # not currently in position
                if z > entry_thres: 
                    if (~np.isnan(t1)) and (~np.isnan(t2)) and (~np.isnan(trade_size)):
                        pos_t1 = -int(trade_size / t1) # enter short position in t1
                        pos_t2 = int(trade_size / t2) # enter long position in t2
                    else:
                        pos_t1 = 0
                        pos_t2 = 0
                elif z < -entry_thres: 
                    if (~np.isnan(t1)) and (~np.isnan(t2)) and (~np.isnan(trade_size)):
                        pos_t1 = int(trade_size / t1) # enter long position in t1
                        pos_t2 = -int(trade_size / t2) # enter short position in t2
                    else:
                        pos_t1 = 0
                        pos_t2 = 0
            else: # currently in a position
                holding_period+=1
                if exit_type == 'fixed':
                    if (abs(z) > entry_thres) & (np.sign(z) == np.sign(pos_t2)):
                        holding_period=0 # reset holding period
                    if holding_period >= exit_thres:
                        # exit positions
                        pos_t1 = 0
                        pos_t2 = 0
                        holding_period = 0
                elif exit_type == 'convergence':
                    if pos_t2 > 0:
                        if z < exit_thres:
                            pos_t1 = 0 # exit short position in t1
                            pos_t2 = 0 # exit long position in t2
                            holding_period = 0
                    elif pos_t2 < 0:
                        if z > -exit_thres:
                            pos_t1 = 0 # exit long position in t1
                            pos_t2 = 0 # exit short position in t2
                            holding_period = 0

            positions_t1.append(pos_t1)
            positions_t2.append(pos_t2)

        prices['positions_' + ticker1] = positions_t1
        prices['positions_' + ticker2] = positions_t2

        prices['pnl_t1'] = np.multiply(np.diff(prices[ticker1]), prices['positions_' + ticker1][1:].shift(1))
        prices['pnl_t2'] = np.multiply(np.diff(prices[ticker2]), prices['positions_' + ticker2][1:].shift(1))
        prices['pnl'] = prices['pnl_t1'] + prices['pnl_t2']
        prices['pnl'] = prices[['ratio_z','pnl']].apply(lambda x: x.pnl if not np.isnan(x.ratio_z) else None, axis=1)

        #prices.to_csv('../data/prices' + '_' + ticker1 + '_' + ticker2 + '.csv')
        return prices['pnl'].dropna()

def compute_max_draw(cumpnl):
    peak = 0
    max_draw = 0
    for d in cumpnl:
        if d:
            draw = d - peak
            if draw < max_draw:
                max_draw = draw
            if d > peak:
                peak = d
    return max_draw