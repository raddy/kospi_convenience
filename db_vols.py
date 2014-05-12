def db_vols(table_name,mysql_con,td,query_type='raw',us_time=False,
    smooth=False,t1=None,t2=None,return_basis=False):
    ignore_cols = ['eventTime','usEventTime','time']
    
    if not t1 and not t2:
        myvols = psql.frame_query('select * from '+table_name,con=mysql_con)
    elif t1 and t2:
        raw_vols = psql.frame_query('select * from '+table_name+' where eventTime>"'+t1+'" and eventTime<"'+t2+'"',con=mysql_con)
    else:
        print 'Must pass zero or no timestamps'
        raise
    
    if myvols is None:
        return [pd.DataFrame(),pd.DataFrame()]
    
    
    if query_type=='raw':
        ignore_cols.append('synthetic')
        ignore_cols.append('futureMidPrice')
        if td>=pd.Timestamp('20131107'): #this is when we start using synthetic offset style
            underlying = (myvols.synthetic + myvols.futureMidPrice)*100
            basis = myvols.synthetic
        else:
            underlying = myvols.synthetic*100
            basis = (myvols.synthetic - myvols.futureMidPrice)*100.0
    else:
        raise 'Only supports raw'
    
    
    myvols['time'] = pd.to_datetime((myvols['eventTime'].apply(str)+'.'+myvols['usEventTime'].apply(str)))

    
    myvols.index = myvols.time
    if us_time:
        myvols.index = myvols.index.tz_localize('America/Chicago').tz_convert('Asia/Seoul')
        
    keep_cols = np.logical_not(pd.Series(myvols.columns).isin(ignore_cols).values)
    
    myvols = myvols.ix[:,keep_cols]
    options = psql.frame_query('select * from options',mysql_sec)


    if td>=pd.Timestamp('20130823'):
        mynames = (pd.Series(myvols.columns).str[1:].astype(int)*10).values
        myvols.columns = mynames
    else:
        options = psql.frame_query('select * from options',mysql_sec)
        ids = pd.Series(myvols.columns).str[1:].astype(int).values
        strike_info = map(lambda x: options[options.id==x].strike.values,ids)
        strikes = map(lambda x: 'skip' if len(x)<1 else str(int(x[0]*100)), strike_info)
        myvols.columns = strikes
        myvols = myvols.ix[:,myvols.columns!='skip']
        myvols = myvols.T.drop_duplicates().T
        
    
    myvols = myvols.fillna(method='ffill').fillna(method='bfill')
    if smooth:
        for c in myvols.columns:
            v = myvols[c]
            v = sm.tsa.filters.hpfilter(v,1e5)[1]
            myvols[c] = v
    if return_basis:
        [myvols,basis]
    return [myvols,underlying]