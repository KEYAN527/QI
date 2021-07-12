# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 02:20:17 2021

@author: Administrator
"""
import pandas as pd
import numpy as np


def get_portfolio_nav(group,broker_name,mkt):    
    portfolio = group[(group['券商'] == broker_name)&(group[mkt+'权重']>0)][[mkt + '基金',mkt + '代码',mkt + '权重']]
    strategy_name = group.loc[group['券商']==broker_name,'策略名称'].values[0]
    time_start = np.str(group[group['券商'] == broker_name]['时间'].values[0])[:10]
    time_end = '2021-07-09'
    del group
    
    portfolio[mkt+'代码'] = portfolio[mkt+'代码'].apply(lambda x:np.str(x)+'.OF')
    fund_list = list(portfolio[mkt+'代码'])
    
    data = pd.read_csv('./data/ETF_nav.csv',encoding = 'gbk')
    data['index'] = pd.to_datetime(data['日期'],format = '%Y-%m-%d')
    data = data.loc[(data['index']>=time_start)&(data['index']<=time_end)&(data['代码'].isin(fund_list)),
                    :]
    nav = data.pivot_table(index = 'index',columns = '代码',values = '复权净值(元)')
    nav = pd.DataFrame(nav.reset_index(),index = range(len(nav))).set_index('index')
    
    #对于策略起始日还没有发布的指数基金产品，以基金基准指数净值作为替代
    index_nav =  pd.read_csv('./data/index_nav.csv',encoding = 'gbk')
    index_nav = index_nav.rename(columns = {'Unnamed: 0':'index'},inplace = False)
    index_nav['index'] = pd.to_datetime(index_nav['index'],format = '%Y-%m-%d')
    for i in nav.columns:
        if nav[i].isnull().values[0] == True:
            i_index = data.loc[data['代码']==i,'标的指数'].values[0]
            if i_index[0] =='h':
                i_index = 'H'+i_index[1:]
            i_indexnav = index_nav.loc[(index_nav['index']>=time_start)&(index_nav['index']<=time_end),i_index]
            i_indexnav = i_indexnav/i_indexnav[:1].values
            nav[i] = i_indexnav.values
    
    for i in nav.columns:
        nav = nav.rename(columns = {
                i:portfolio.loc[portfolio[mkt+'代码']== i,mkt+'基金'].values[0]},
                inplace = False)
#    nav = nav.fillna(1)
    nav = nav/nav[:1].values
    
    weight = list(portfolio[mkt+'权重'])
    
    port_nav = np.dot(nav,weight)
    port_nav = pd.DataFrame(port_nav,columns = [strategy_name],index = nav.index)
    nav = pd.merge(nav.reset_index(),port_nav.reset_index(),on = ['index'],how = 'outer')
    nav = nav.set_index(['index'])
    del port_nav
    del portfolio
    
    return nav 

def get_index(nav,index_code = '000905.XSHG'):
    daily_index = pd.read_csv('./data/bench_index_nav.CSV')
    daily_index['index'] = pd.to_datetime(daily_index['index'],format = '%Y-%m-%d')
    
    daily_index = daily_index.loc[(daily_index['index']>=nav.index.values[0])&(daily_index['index']<=nav.index.values[-1]),
                                  ['index',index_code]]
    daily_index = daily_index.sort_values(['index'])
    daily_index[index_code] = daily_index[index_code]/daily_index[index_code][:1].values[0]
    
    nav = nav.reset_index()
    nav['index'] = pd.to_datetime(nav['index'],format = '%Y-%m-%d')
    nav = pd.merge(nav,daily_index,on = ['index'],how = 'left')
    nav = nav.set_index('index')
    del daily_index
    
    return nav

def maxdrop(dataset):
    cum = np.cumprod(1+dataset)
    drop = []
    for k in range(len(cum)):
        maxd = cum[k:k+1].values/(cum[:k+1].max()) - 1
        drop = np.append(drop,maxd)
    mdd = min(drop)
    return mdd

def get_perf(port_nav,port_name):
    perf = pd.DataFrame(columns = {'收益率-1M','收益率-3M',
                                   '收益率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                                   '年化波动率-1M','年化波动率-3M',
                                   '年化波动率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                                   '年化夏普-1M','年化夏普-3M',
                                   '年化夏普-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                                   '最大回撤-1M','最大回撤-3M',
                                   '最大回撤-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'},
        index = [port_name])

    nav_s = port_nav.reset_index()
    nav_s = nav_s.loc[:,['index',port_name]]
    nav_s = nav_s.sort_values(['index'])
    nav_s[port_name] = nav_s[port_name]/nav_s[port_name].shift(1) - 1
    nav_s = nav_s[1:]
    nav_s = nav_s.reset_index(drop = True)
    
    nav_1 = nav_s[-21:]
    perf['收益率-1M'] = [np.cumprod(1+nav_1[port_name]).values[-1]-1]
    perf['年化波动率-1M'] = [np.std(nav_1[port_name])*(252**0.5)]
    perf['年化夏普-1M'] = [((perf['收益率-1M'].values[0]+1)**(252/21)-1-0.02)/perf['年化波动率-1M'].values[0]]
    perf['最大回撤-1M'] = maxdrop(nav_1[port_name])
    del nav_1

    nav_3 = nav_s[-63:]
    perf['收益率-3M'] = [np.cumprod(1+nav_3[port_name]).values[-1]-1]
    perf['年化波动率-3M'] = [np.std(nav_3[port_name])*(252**0.5)]
    perf['年化夏普-3M'] = [((perf['收益率-3M'].values[0]+1)**(252/63)-1-0.02)/perf['年化波动率-3M'].values[0]]
    perf['最大回撤-3M'] = maxdrop(nav_3[port_name])
    del nav_3

    perf['收益率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'] = [np.cumprod(1+nav_s[port_name]).values[-1]-1]
    perf['年化波动率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'] = [np.std(nav_s[port_name])*(252**0.5)]
    perf['年化夏普-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'] = [
            ((perf['收益率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'].values[0]+1)**(252/len(nav_s))-1-0.02)/
            perf['年化波动率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'].values[0]]
    perf['最大回撤-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今'] = maxdrop(nav_s[port_name])
    
    perf = np.round(perf,4)
    
    for k in ['收益率-1M','收益率-3M','收益率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                '年化波动率-1M','年化波动率-3M','年化波动率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                '最大回撤-1M','最大回撤-3M','最大回撤-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今']:
        perf[k] = perf[k].apply(lambda x:"%.2f%%" % (x * 100))
    
    perf = perf[['收益率-1M','收益率-3M','收益率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                '年化波动率-1M','年化波动率-3M','年化波动率-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                '年化夏普-1M','年化夏普-3M','年化夏普-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今',
                '最大回撤-1M','最大回撤-3M','最大回撤-'+port_nav.index[0].strftime('%Y/%m/%d')+'至今']]
    del port_nav
    del nav_s
    
    return perf

def get_ETFindex_perf(group,broker_name,mkt,last_rpt_period):
    portfolio = group[(group['券商'] == broker_name)&(group[mkt+'权重']>0)][[mkt + '基金',mkt + '代码',mkt + '权重']]
    del group
    
    portfolio[mkt+'代码'] = portfolio[mkt+'代码'].apply(lambda x:np.str(x)+'.OF')
    fund_list = list(portfolio[mkt+'代码'])
    
    ETF = pd.read_excel('./data/权益类ETF产品列表.xlsx')
    ETF = ETF[ETF['基金代码'].isin(fund_list)]
    ETF = ETF[['基金代码','PB(LY)','PE(TTM)','归母净利润同比增长率']]
    ETF = ETF.rename(columns = {'基金代码':mkt+'代码'},inplace = False)
    data = pd.merge(ETF,portfolio,on = [mkt+'代码'],how = 'outer')
    del ETF
    data = data[[mkt+'基金',mkt+'代码',mkt+'权重','PB(LY)','PE(TTM)','归母净利润同比增长率']]
    del portfolio
    
    return data
    

def get_holding(group,broker_name,mkt):
    portfolio = group[(group['券商'] == broker_name)&(group[mkt+'权重']>0)][[mkt + '基金',mkt + '代码',mkt + '权重']]
    portfolio[mkt+'代码'] = portfolio[mkt+'代码'].apply(lambda x:np.str(x)+'.OF')
    fund_list = list(portfolio[mkt+'代码'])
    
    ETF = pd.read_excel('./data/权益类ETF产品列表.xlsx')
    index = ETF.loc[ETF['基金代码'].isin(fund_list),['基金代码','标的指数']]
    del ETF
    
    index_consti = pd.read_excel('./data/指数-持仓权重.xlsx')
    index_consti = index_consti[index_consti['指数代码'].isin(list(index['标的指数']))]
    
    for i in list(index['标的指数']):
        ETF_code = index.loc[index['标的指数']==i,'基金代码'].values[0]
        index_consti.loc[index_consti['指数代码']==i,'weight'] = index_consti.loc[index_consti['指数代码']==i,'weight'
                         ]*portfolio.loc[portfolio[mkt+'代码']==ETF_code,mkt+'权重'].values[0]
    del index
    
    stk = index_consti.groupby(['股票代码'])['weight'].sum().reset_index()
    del index_consti
#    print('组合成分股权重加和等于：',np.round(top_stk['weight'].sum(),4))
    top_stk = stk.sort_values(['weight'],ascending = False)[:20]
    top_stk = top_stk.reset_index(drop = True)
#    print('仓位前20股票权重之和等于',"%.2f%%" % (top_stk['weight'].sum()*100))
    
    stk_l = list(set(top_stk['股票代码']))
    df = pd.read_excel('./data/指数全部持仓_标签汇总.xlsx',sheet_name = 'Sheet1')
    df = df[df['股票代码'].isin(stk_l)]
    
    top_stk = top_stk.merge(df[['股票代码','股票名称','财报主营构成-项目名称','区间收益率']],on = ['股票代码'],how = 'outer')    
    top_stk = top_stk.sort_values(['weight'],ascending = False)
    top_stk = top_stk.reset_index(drop = True)

    top_stk['weight'] = top_stk['weight'].apply(lambda x:"%.2f%%" % (x * 100))
    top_stk['区间收益率'] = top_stk['区间收益率'].apply(lambda x:"%.2f%%" % x)
    top_stk = top_stk.rename(columns = {'weight':'股票权重'
                                        ,'财报主营构成-项目名称':'财报主营构成'},inplace = False)
    
    label = pd.merge(stk[['股票代码','weight']],df[['股票代码','一级标签','申万行业明细']].drop_duplicates(),
                         on = ['股票代码'],how = 'left')
    del df
    del stk
    label = label.rename(columns = {'weight':'股票权重'},inplace = False)
    label['股票权重'] = label['股票权重']/label['股票权重'].sum()
    stk_label = label.groupby(label['一级标签'])['股票权重'].sum().reset_index()
    stk_label = stk_label.sort_values(['股票权重'],ascending = False)
    stk_label['股票权重'] = stk_label['股票权重'].apply(lambda x:"%.2f%%" % (x*100))
    stk_label = stk_label.rename(columns = {'股票权重':'标签占比'},inplace = False)
    
    port_ind = label.groupby(['申万行业明细'])['股票权重'].sum().reset_index()
    port_ind = port_ind.sort_values(['股票权重'],ascending = False)
    port_ind = port_ind.reset_index(drop = True)
    port_ind['股票权重'] = port_ind['股票权重'].apply(lambda x:"%.2f%%" % (x * 100))
    
    del label
    
    return top_stk,port_ind,stk_label




