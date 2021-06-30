# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 02:20:17 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from WindPy import *
w.start()

#作图函数
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "none"

import jqdatasdk as jq
# 聚宽账号
jq.auth('13683364770', '1113Rorschach')
# 查询是否连接成功
is_auth = jq.is_auth()
print(is_auth)

# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:25:41 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from WindPy import *
w.start()

#作图函数
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "none"

import jqdatasdk as jq
# 聚宽账号
jq.auth('13683364770', '1113Rorschach')
# 查询是否连接成功
is_auth = jq.is_auth()
print(is_auth)

def get_portfolio_nav(group,broker_name,mkt):    
    portfolio = group[(group['券商'] == broker_name)&(group[mkt+'权重']>0)][[mkt + '基金',mkt + '代码',mkt + '权重']]
    portfolio[mkt+'代码'] = portfolio[mkt+'代码'].apply(lambda x:np.str(x)+'.OF')
    strategy_name = group.loc[group['券商']==broker_name,'策略名称'].values[0]
    fund_list = list(portfolio[mkt+'代码'])

    time_start = np.str(group[group['券商'] == broker_name]['时间'].values[0])[:10]
    time_end = datetime.now().strftime('%Y-%m-%d')
    
    data = w.wsd(",".join(fund_list), "NAV_adj", time_start, time_end, "")
    nav = pd.DataFrame(data.Data).T
    nav.columns = data.Codes
    nav.index = data.Times
    #对于策略起始日还没有发布的指数基金产品，以基金基准指数净值作为替代
    for i in nav.columns:
        if nav[i].isnull().values[0] == True:
            i_index = w.wsd(i,"fund_trackindexcode").Data[0][0]
            i_indexnav = pd.DataFrame(w.wsd(i_index, "close", time_start, time_end, "").Data[0])
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
    
    return nav 

def get_index(nav,index_code = '000905.XSHG'):
    daily_index = jq.get_price(index_code, start_date=np.str(nav.index.values[0])[:10], 
                               end_date= np.str(nav.index.values[-1])[:10],
                               frequency='daily',
                               fields=None, 
                               skip_paused=False, fq=None, panel=False, fill_paused=False)
    daily_index = daily_index[['close']]
    daily_index = daily_index/daily_index[:1].values
    daily_index = daily_index.reset_index()
    daily_index = daily_index.sort_values(['index'])
    daily_index = daily_index.rename(columns = {'close':index_code},inplace = False)
    daily_index['index'] = pd.to_datetime(daily_index['index'],format = '%Y-%m-%d')
    
    nav = nav.reset_index()
    nav['index'] = pd.to_datetime(nav['index'],format = '%Y-%m-%d')
    nav = pd.merge(nav,daily_index,on = ['index'],how = 'left')
    nav = nav.set_index('index')
    
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

    nav_3 = nav_s[-63:]
    perf['收益率-3M'] = [np.cumprod(1+nav_3[port_name]).values[-1]-1]
    perf['年化波动率-3M'] = [np.std(nav_3[port_name])*(252**0.5)]
    perf['年化夏普-3M'] = [((perf['收益率-3M'].values[0]+1)**(252/63)-1-0.02)/perf['年化波动率-3M'].values[0]]
    perf['最大回撤-3M'] = maxdrop(nav_3[port_name])

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
    
    return perf


def get_ETFindex_perf(group,broker_name,mkt,last_rpt_period):
    portfolio = group[(group['券商'] == broker_name)&(group[mkt+'权重']>0)][[mkt + '基金',mkt + '代码',mkt + '权重']]
    portfolio[mkt+'代码'] = portfolio[mkt+'代码'].apply(lambda x:np.str(x)+'.OF')
    fund_list = list(portfolio[mkt+'代码'])

    time_end = datetime.now().strftime('%Y-%m-%d')
    
    index_list = pd.DataFrame(w.wsd(",".join(fund_list),"fund_trackindexcode").Data,
                              columns = w.wsd(",".join(fund_list),"fund_trackindexcode").Codes)
    
    index_perf = w.wss(",".join(list(index_list.T[0])),"pe_ttm,,pb_lf,yoy_or,yoynetprofit","tradeDate="+time_end+
                                     ";rptDate="+last_rpt_period)
    index_perf_table = pd.DataFrame( index_perf.Data,index = index_perf.Fields,columns = index_perf.Codes)
    index_perf_table = np.round(index_perf_table,2)
    index_perf_table = index_perf_table.T.reset_index()
    index_perf_table = index_perf_table.rename(columns = {'PE_TTM':'市盈率PE_TTM',
                                                          'PB_LF':'市净率PB',
                                                          'YOY_OR':'营业收入同比增长率（%）',
                                                          'YOYNETPROFIT':'归母净利润同比增长率（%）'},
                                             inplace = False)
    
    index_list = index_list.T.reset_index()
    index_list = index_list.rename(columns = {'index':mkt+'代码'},inplace = False)
    index_list = index_list.rename(columns = {0:'index'},inplace = False)
    index_list['index'] = index_list['index'].apply(lambda x:x.upper())
    
    index_perf_table = pd.merge(index_perf_table,index_list,on = ['index'],how = 'outer')
    index_perf_table = pd.merge(index_perf_table,portfolio,on = [mkt+'代码'],how = 'outer')
    index_perf_table = index_perf_table[[mkt+'基金',mkt+'代码',mkt+'权重','市盈率PE_TTM','市净率PB',
                                         '营业收入同比增长率（%）','归母净利润同比增长率（%）']]
    return index_perf_table
    

def get_holding(group,broker_name,mkt):
    portfolio = group[(group['券商'] == broker_name)&(group[mkt+'权重']>0)][[mkt + '基金',mkt + '代码',mkt + '权重']]
    portfolio[mkt+'代码'] = portfolio[mkt+'代码'].apply(lambda x:np.str(x)+'.OF')
    fund_list = list(portfolio[mkt+'代码'])

    time_start = np.str(group[group['券商'] == broker_name]['时间'].values[0])[:10]
    time_end = datetime.now().strftime('%Y-%m-%d')
    
    index_list = pd.DataFrame(w.wsd(",".join(fund_list),"fund_trackindexcode").Data,
                              columns = w.wsd(",".join(fund_list),"fund_trackindexcode").Codes)
    
    index_consti = pd.DataFrame()
    for k in index_list.columns:
        k_consti = pd.DataFrame(w.wset("indexconstituent","date="+time_end+";windcode=" + index_list[k].values[0]).Data).T
        k_consti.columns = ['date','stk','name','weight','industry']
        k_consti['weight'] = k_consti['weight']/100*(portfolio.loc[portfolio[mkt+'代码']==k,mkt+'权重'].values[0])
        
        index_consti = pd.concat([index_consti,k_consti],sort = True)
        
    top_stk = index_consti.groupby(['stk'])['weight'].sum().reset_index()
    print('组合成分股权重加和等于：',np.round(top_stk['weight'].sum(),4))
    top_stk = top_stk.sort_values(['weight'],ascending = False)[:20]
    top_stk = top_stk.reset_index(drop = True)
    print('仓位前20股票权重之和等于',"%.2f%%" % (top_stk['weight'].sum()*100))

    
    stk_l = list(set(top_stk['stk']))
    df = pd.DataFrame(
            w.wss(",".join(stk_l), "sec_name,segment_product_item,pct_chg_per",
                  "rptDate="+np.str(np.float(time_end[:4])-1)[:4]+"1231;order=1;startDate="+
                  time_start[:4]+time_start[5:7]+time_start[8:10]+
                  ";endDate="+time_end[:4]+time_end[5:7]+time_end[8:10]).Data).T
    df.columns = ['股票名称','财报主营构成-项目名称','区间涨跌幅']
    df['stk'] = stk_l
    
    top_stk = top_stk.merge(df,on = ['stk'],how = 'outer')    
    top_stk = top_stk.sort_values(['weight'],ascending = False)
    top_stk = top_stk.reset_index(drop = True)

    top_stk['weight'] = top_stk['weight'].apply(lambda x:"%.2f%%" % (x * 100))
    top_stk['区间涨跌幅'] = top_stk['区间涨跌幅'].apply(lambda x:"%.2f%%" % x)
    top_stk = top_stk.rename(columns = {'stk':'股票代码','weight':'股票权重'
                                        ,'财报主营构成-项目名称':'财报主营构成'},inplace = False)
    
    stk_label = pd.read_excel('./data/指数重仓股票_标签汇总.xlsx',sheet_name = 'Sheet1')
    stk_label = pd.merge(top_stk[['股票代码','股票权重']],stk_label[['股票代码','一级标签']].drop_duplicates(),
                         on = ['股票代码'],how = 'left')
    stk_label['股票权重'] = stk_label['股票权重'].apply(lambda x:np.float(x[:-1]))
    stk_label['股票权重'] = stk_label['股票权重']/stk_label['股票权重'].sum()
    stk_label = stk_label.groupby(stk_label['一级标签'])['股票权重'].sum().reset_index()
    stk_label = stk_label.sort_values(['股票权重'],ascending = False)
    stk_label['股票权重'] = stk_label['股票权重'].apply(lambda x:"%.2f%%" % (x*100))
    stk_label = stk_label.rename(columns = {'股票权重':'标签占比'},inplace = False)
    
    port_ind = index_consti.groupby(['industry'])['weight'].sum().reset_index()
    port_ind = port_ind.sort_values(['weight'],ascending = False)
    port_ind = port_ind.reset_index(drop = True)
    port_ind['weight'] = port_ind['weight'].apply(lambda x:"%.2f%%" % (x * 100))
    port_ind = port_ind.rename(columns = {'industry':'行业分类','weight':'股票权重'},inplace = False)
    
    return top_stk,port_ind,stk_label




