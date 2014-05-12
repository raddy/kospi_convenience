#KOPSI CONVENIENCE FUNCTIONS
import pandas as pd
import numpy as np
from cycross import cross,fix_timestamps
from krx_normalize import two_level_fix_a3s
from bsm_implieds import net_effect_cython
import MySQLdb as sql
import pandas.io.sql as psql
import statsmodels.api as sm
import MySQLdb

expiry_info = pd.Series([])

def load_expiries(expiry_path):
    df = pd.read_csv(expiry_path,header=None)
    df.columns = ['code','expiry']
    df.expiry =  pd.to_datetime(df.expiry,format='%Y%m%dT%H%M%S')
    expiry_info = pd.Series(df['code'].values,index=df.expiry)
    return pd.Series(df['code'].values,index=df.expiry)


def get_dte(expiry,trade_date):
    drange = pd.date_range(end=pd.Timestamp(expiry),start=pd.Timestamp(trade_date))
    s = (pd.Series(pd.to_datetime(drange).tolist(),index=drange).asfreq(pd.tseries.offsets.BDay()))
    return len(s)

def get_expiry(trade_date):
    return expiry_info[(expiry_info.index>=pd.Timestamp(trade_date))].index[0]

def get_expiry_code(trade_date):
    return expiry_info[(expiry_info.index>=pd.Timestamp(trade_date))].values[0]

def front_underlying(some_md):
    s = pd.Series(some_md.symbol.value_counts().index.values)
    return s[(s.str[2:6]=='4101').values].values[0]

def underlying_code_by_two_digit_code(code):
    single_codes = ['1','2','3','4','5','6','7','8','9','A','B','C']
    return str(code[0])+dict(zip(single_codes,np.repeat(['3','6','9','C'],3).tolist()))[code[1]]

def underlyings(some_md):
    s = pd.Series(some_md.symbol.value_counts().index.values)
    return s[s.str[2:4]=='41']


def underlying_code(exp_code,u):
    return u[u.str[6:8]==exp_code]

def front_code(some_dict_of_expirys):
    return sorted(some_dict_of_expirys,key=some_dict_of_expirys.get)[0]

def deio_code_to_krx_code(deio_code):
    year = int(deio_code[-2:])
    
    if year == 13:
        year_code = 'H'
    elif year == 14:
        year_code = 'J'
    else:
        print 'Unsupported Year'
        raise Exception
    
    single_codes = np.array(['1','2','3','4','5','6','7','8','9','A','B','C'])
    us_codes = np.array(['F','G','H','J','K','M','N','Q','U','V','X','Z'])
    
    month_code = single_codes[us_codes==deio_code[3]]
    if len(month_code)!=1:
        print 'Bad Month'
        raise Exception
        
    return year_code+month_code[0]

#nearby strike mask for put option symbols
def nearby_puts_mask(futures_price,strike_array,symbol_list,strike_width):
    nearby_strike = abs(strike_array-futures_price)<strike_width
    is_puts = pd.Series(symbol_list).str[2:4]=='43'
    return np.logical_and(nearby_strike,is_puts)

#nearby strike mask for call option symbols
def nearby_calls_mask(futures_price,strike_array,symbol_list,strike_width):
    nearby_strike = abs(strike_array-futures_price)<strike_width
    is_call = pd.Series(symbol_list).str[2:4]=='42'
    return np.logical_and(nearby_strike,is_call)

#takes a LIST/ARRAY of KRX Kospi symbols -- returns a list of float strikes
def kospi_strikes_from_symbols(symbol_list):
    strikes = pd.Series(symbol_list).str[8:11].astype(float)
    strikes[strikes%5 != 0 ] += .5
    return strikes * 100

#takes an expiry code (2 digit) plus a LIST/ARRAY of KRX Kospi symbols -- returns a pandas series
#  -- said panda series is a list of option issue codes that are both options and of matching expiry
def option_symbols(expiry_code,symbol_list):
    s = pd.Series(symbol_list)
    are_options = np.logical_or(s.str[2:4]=='43',s.str[2:4]=='42')
    matching_code = s.str[6:8]==expiry_code
    return s[np.logical_and(are_options,matching_code)]

def is_kospi(symbol_list):
    s = pd.Series(symbol_list)
    is_call = s.str[2:6]=='4201'
    is_put = s.str[2:6]=='4301'
    is_fut = s.str[2:6]=='4101'
    return np.logical_or(is_call,np.logical_or(is_put,is_fut))

def kospi_types_from_symbols(symbol_list):
    res = np.zeros(len(symbol_list))
    is_call = pd.Series(symbol_list).str[2:4]=='42'
    is_put = pd.Series(symbol_list).str[2:4]=='43'
    res[is_call.values] = 1
    res[is_put.values] = -1
    return res

def kospi_strikes_from_symbols(symbol_list):
    strikes = pd.Series(symbol_list).str[8:11].astype(float)
    strikes[strikes%5 != 0 ] += .5
    return (strikes * 100).values

def kospi_fresh(symbol_list,tte_list,age):
    s = pd.Series(symbol_list)
    is_call = s.str[2:6]=='4201'
    is_put = s.str[2:6]=='4301'
    is_fut = s.str[2:6]=='4101'
    is_option = np.logical_or(is_call,is_put)
    fresh_options = np.logical_and(is_option,tte_list<age)
    return np.logical_or(is_fut,fresh_options)   
     
def options_expiry_mask(symbol_list,expiry_code,include_futs=True):
    s = pd.Series(symbol_list)
    is_call = s.str[2:6]=='4201'
    is_put = s.str[2:6]=='4301'
    is_fut = s.str[2:6]=='4101'
    is_option = np.logical_or(is_call,is_put)
    is_expiry = s.str[6:8]==expiry_code
    if not include_futs:
        return np.logical_and(is_option,is_expiry)
    return np.logical_and(np.logical_or(is_option,is_fut),is_expiry)

def to_sql_time(a_pandas_timestamp):
    return pd.Timestamp(a_pandas_timestamp).strftime('%Y-%m-%d %H:%M:%S')

def to_sql_date(a_pandas_timestamp):
    return pd.Timestamp(a_pandas_timestamp).strftime('%Y-%m-%d').replace('-','')

def to_expiry_stamp(a_pandas_timestamp):
    return pd.Timestamp(a_pandas_timestamp.strftime('%Y-%m-%d')+'T14:50:00')

def fetch_front_h5_rt_vols(fn,two_digit_code,expiry):
    
    h5_pointer = pd.HDFStore(fn)
    pcap_info = h5_pointer['pcap_data']

    mysql_con = sql.connect(host='10.1.31.202',
                port=3306,user='deio', passwd='!w3alth!',
                db='DEIO_SY')

    my_und = underlying_code(underlying_code_by_two_digit_code(two_digit_code),underlyings(pcap_info)).values[0]
    just_that_data = pcap_info[options_expiry_mask(pcap_info.symbol,two_digit_code)]
    
    fixed_data = two_level_fix_a3s(just_that_data.symbol.values,just_that_data.msg_type.str[1:].astype(long).values,just_that_data.ix[:,['bid1','bidsize1','ask1','asksize1','bid2','bidsize2','ask2','asksize2','tradeprice','tradesize']].values)
    just_that_fut = pcap_info[pcap_info.symbol==my_und]
    strikes = np.array(kospi_strikes_from_symbols(just_that_data.symbol.values).astype(int),dtype=object)

    start_time,end_time = fn.replace('.h5','').split('/')[-1].split('_')
    sql_t1,sql_t2 = to_sql_time(start_time),to_sql_time(end_time)
    sql_date = to_sql_date(start_time)
    dte = get_dte(expiry,sql_date)
    table_name = 'opm_'+sql_date+'_'+get_expiry_code(sql_t1)+'_smoothedVols'
    vols,basis = db_vols(table_name,mysql_con,pd.Timestamp(sql_date),query_type='raw',
        us_time=False,smooth=False,t1=sql_t1,t2=sql_t2,return_basis=True)
    vols.index = vols.index.astype(np.int64)
    basis.index = vols.index
    
    c = cross(strikes,just_that_data.index.astype(long),vols.index.values,vols.columns.values.astype(object),vols.values)

    just_that_data['tte'] = dte/260.
    just_that_data['vols'] = c
    

    
    expiries = just_that_data.symbol.str[6:8]    
    just_that_data['basis'] = basis.asof(just_that_data.index).fillna(method='ffill')
    just_that_data['strike'] = strikes
    just_that_data['type'] = kospi_types_from_symbols(just_that_data.symbol.values)
    dat = just_that_data.append(just_that_fut).sort_index()
    dat.index = pd.to_datetime(dat.index)
    dat.basis = dat.basis.fillna(method='ffill')
        
    #throw out 'expired' options data
    if dte<=1:
        new_end_time = to_expiry_stamp(pd.Timestamp(start_time))
        dat = dat.ix[:new_end_time]

    return dat

def fetch_front_h5(fn,two_digit_code,expiry):
    
    h5_pointer = pd.HDFStore(fn)
    pcap_info = h5_pointer['pcap_data']
    
    pcap_info.index = fix_timestamps(pcap_info.index.values)
    
    k = '/'+two_digit_code+'/vols'
    if k not in h5_pointer.keys():
        print('ERROR: KEY NOT IN H5!')
        raise Exception
    

    my_und = underlying_code(underlying_code_by_two_digit_code(two_digit_code),underlyings(pcap_info)).values[0]
    just_that_data = pcap_info[options_expiry_mask(pcap_info.symbol,two_digit_code)]
    
    fixed_data = two_level_fix_a3s(just_that_data.symbol.values,just_that_data.msg_type.str[1:].astype(long).values,just_that_data.ix[:,['bid1','bidsize1','ask1','asksize1','bid2','bidsize2','ask2','asksize2','tradeprice','tradesize']].values)

    just_that_fut = pcap_info[pcap_info.symbol==my_und]
    strikes = np.array(kospi_strikes_from_symbols(just_that_data.symbol.values).astype(int),dtype=object)
    just_those_vols = h5_pointer[k]
    basis = h5_pointer['basis'][two_digit_code]
    basis.index = basis.index.astype(np.int64)
    just_that_data['tte'] = just_that_data.symbol.str[6:8].replace(h5_pointer['dtes'].to_dict()[0]).astype(float) / 260.
    just_that_data['vols'] = cross(strikes,just_that_data.index.astype(long),just_those_vols.index.astype(long),just_those_vols.columns.values,just_those_vols.values)
    expiries = just_that_data.symbol.str[6:8]    
    just_that_data['basis'] = basis.asof(just_that_data.index).fillna(method='ffill')
    just_that_data['strike'] = strikes
    just_that_data['type'] = kospi_types_from_symbols(just_that_data.symbol.values)
    dat = just_that_data.append(just_that_fut).sort_index()
    dat.index = pd.to_datetime(dat.index)
    dat.basis = dat.basis.fillna(method='ffill')
    
    start_time,end_time = fn.replace('.h5','').split('/')[-1].split('_')
    dte = get_dte(expiry,start_time)

    #throw out 'expired' options data
    if dte<=1:
        new_end_time = to_expiry_stamp(pd.Timestamp(start_time))
        dat = dat.ix[:new_end_time]
    return dat

def kospi_clean_futs(fn):
    store = pd.HDFStore(fn)
    dat = store['pcap_data']
    dat.index = fix_timestamps(dat.index.values)
    store.close()
    front_fut = front_underlying(dat)
    futs = dat[dat.symbol==front_fut]
    futs = futs[futs.msg_type!='B2']
    futs.index = pd.to_datetime(futs.index)
    nfuts = pd.DataFrame(two_level_fix_a3s(futs.symbol.values,futs.msg_type.str[1:].astype(long).values,futs.ix[:,['bid1','bidsize1','ask1','asksize1','bid2','bidsize2','ask2','asksize2','tradeprice','tradesize']].values),columns = ['bid1','bidsize1','ask1','asksize1','bid2','bidsize2','ask2','asksize2','tradeprice','tradesize'], index = futs.index)
    nfuts.tradesize = pd.Series(net_effect_cython(futs.symbol.values,nfuts.bid1.astype(np.double).values,nfuts.tradeprice.astype(np.double).values,nfuts.tradesize.values)).replace(np.NaN,0).values.astype(long)
    return nfuts[np.logical_not(nfuts.sum(axis=1)==0)]

def deio_strat_id(mysql_con,strategy):
    types_info = psql.frame_query("""select * from DEIO_META.orderTypes""", con=mysql_con)
    types_dict =  dict(zip(types_info.name,types_info.orderTypeId))
    return types_dict[strategy]

def deio_user_id(mysql_con,who):
    user_info = psql.frame_query("""select * from DEIO_META.users""", con=mysql_con)
    user_dict = dict(zip(user_info.userName,user_info.userId))
    return user_dict[who]
    
def db_orders(file_name,user=None,strat=None):
    
    start_time,end_time = file_name.replace('.h5','').split('/')[-1].split('_')
    t1,t2 = to_sql_time(start_time),to_sql_time(end_time)

    mysql_con = sql.connect(host='10.1.31.202',
                port=3306,user='deio', passwd='!w3alth!',
                db='DEIO_SY')

    if user and strat:
        userid = str(deio_user_id(mysql_con,user))
        typeid = str(deio_strat_id(mysql_con,strat))
        ords = psql.frame_query('select id,orderStatus,qty,price,side,eventTime,usEventTime,exchOrderId,triggerId from orders where eventTime>"'+t1+'" and eventTime<"'+t2+
                                '" and userId="'+userid+'" and type='+typeid +' and price<20',con=mysql_con)
        ords['time'] = pd.to_datetime((ords['eventTime'].apply(str)+'.'+ords['usEventTime'].apply(str)))
        ords['triggerId'] = ords.triggerId % 1e6
        ords.index = ords.time
        def funky_town(x):
            x['Filled'] = np.logical_or(x.orderStatus==7,x.orderStatus==8).max()
            return x
        nords = ords.groupby('id').apply(lambda x: funky_town(x).sort_index(by='time')).groupby(level=0).first()
        nords.index = nords.time
        return nords
    else:
        print 'Only user + strat supported'
        raise

def db_ords(start_time,end_time):

    t1,t2 = to_sql_time(start_time),to_sql_time(end_time)
    mysql_con = sql.connect(host='10.1.31.202',
                port=3306,user='deio', passwd='!w3alth!',
                db='DEIO_SY')

    ords = psql.frame_query('select id,orderStatus,qty,price,side,eventTime,usEventTime,exchOrderId,triggerId from orders where eventTime>"'+t1+'" and eventTime<"'+t2+
                                '"+',con=mysql_con)

def to_sql_time(some_timestamp):
    return some_timestamp.strftime('%Y-%m-%d %H:%M:%S')

def db_rt_vols(mysql_con,t1,t2):


    sql_t1,sql_t2 = to_sql_time(t1),to_sql_time(t2)
    td = t1.strftime("%Y%m%d")

    expiry = get_expiry(td)
    dte = get_dte(expiry,td)
    table_name = 'opm_'+td+'_'+get_expiry_code(td)+'_smoothedVols'


    vols,basis = db_vols(table_name,mysql_con,pd.Timestamp(td),query_type='raw',
        us_time=False,smooth=False,t1=sql_t1,t2=sql_t2,return_basis=True)
    vols.index = vols.index.astype(np.int64)
    basis.index = vols.index
    return [vols,basis]

def db_vols(table_name,mysql_con,td,query_type='raw',us_time=False,
    smooth=False,t1=None,t2=None,return_basis=False):
    
    ignore_cols = ['eventTime','usEventTime','time']

    if 'smoothed' in table_name:
        raw_table_name = table_name.replace('smoothed','raw')
    else:
        raw_table_name = table_name

    if not t1 and not t2:
        rawvols = psql.frame_query('select * from '+raw_table_name,con=mysql_con)
    elif t1 and t2:
        rawvols = psql.frame_query('select * from '+raw_table_name+' where eventTime>"'+t1+'" and eventTime<"'+t2+'"',con=mysql_con)
    else:
        print 'Must pass zero or no timestamps'
        raise
    
    if rawvols is None:
        return [pd.DataFrame(),pd.DataFrame()]
    
    if td>=pd.Timestamp('20131107'): #this is when we start using synthetic offset style
        underlying = (rawvols.synthetic + rawvols.futureMidPrice)*100
        basis = rawvols.synthetic * 100
    else:
        underlying = rawvols.synthetic*100
        basis = (rawvols.synthetic - rawvols.futureMidPrice)*100.0

    
    if query_type=='raw':
        ignore_cols.append('synthetic')
        ignore_cols.append('futureMidPrice')
        myvols = rawvols
    else:
        if not t1 and not t2:
            nvols = psql.frame_query('select * from '+table_name,con=mysql_con)
            myvols = nvols.head(len(rawvols))
        elif t1 and t2:
            nvols = psql.frame_query('select * from '+table_name+' where eventTime>"'+t1+'" and eventTime<"'+t2+'"',con=mysql_con)
            myvols = nvols.head(len(rawvols))
        else:
            print 'Must pass zero or no timestamps'
            raise
    
    myvols['time'] = pd.to_datetime((myvols['eventTime'].apply(str)+'.'+myvols['usEventTime'].apply(str)))
    myvols.index = myvols.time
    basis.index = rawvols.index

    if us_time:
        myvols.index = myvols.index.tz_localize('America/Chicago').tz_convert('Asia/Seoul')
        
    keep_cols = np.logical_not(pd.Series(myvols.columns).isin(ignore_cols).values)
    
    myvols = myvols.ix[:,keep_cols]
    


    if td>=pd.Timestamp('20130823'):
        mynames = (pd.Series(myvols.columns).str[1:].astype(int)*10).values
        myvols.columns = mynames
    else:
        mysql_sec = MySQLdb.connect(host='10.1.31.202', 
                port=3306,user='deio', passwd='!w3alth!', 
                db='SECURITIES')
        options = psql.frame_query('select * from options',mysql_sec)
        ids = pd.Series(myvols.columns).str[1:].astype(int).values
        strike_info = map(lambda x: options[options.id==x].strike.values,ids)
        strikes = map(lambda x: 'skip' if len(x)<1 else str(int(x[0]*100)), strike_info)
        myvols.columns = strikes
        myvols = myvols.ix[:,myvols.columns!='skip']
        myvols = myvols.T.drop_duplicates().T
        
    
    myvols = myvols.fillna(method='ffill').fillna(method='bfill').dropna(axis=1,how='all')
    basis = basis.fillna(method='ffill').fillna(method='bfill')
    if smooth:
        for c in myvols.columns:
            v = myvols[c]
            v = sm.tsa.filters.hpfilter(v,1e5)[1]
            myvols[c] = v
    if return_basis:
        return [myvols,basis]
    else:
        return [myvols,underlying]
