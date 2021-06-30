# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:29:52 2021

@author: Administrator
"""
import os 
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import scipy.stats
# import plotly.express as px
import plotly.graph_objects as go
# import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.subplots import make_subplots
import seaborn as sns
from matplotlib import pyplot as plt

import matplotlib.dates as mdate
from matplotlib.pyplot import MultipleLocator
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pyecharts.charts import Bar, Pie,Line
#用于设值全局配置和系列配置（二者的区别已经在柱状图的博文中讲解过）
from pyecharts import options as opts

# Import the hypothesis_testing.py module
#from hypothesis_testing import *
#import streamlit_analytics
import sys
sys.path.append("D:\K\project_indexfund\demo\script")
from ETF_funcs import *

class port_ana(object):
    def __init__(self, group,broker_name,mkt,index_list,last_rpt_period):
        self.group = group
        self.broker_name = broker_name
        self.mkt = mkt
        self.index_list = index_list
        self.last_rpt_period = last_rpt_period
    
    def get_nav(self):
        self.strategy_name = self.group.loc[self.group['券商']==self.broker_name,'策略名称'].values[0]
        self.port_nav = get_portfolio_nav(self.group,self.broker_name,self.mkt)
        for i in self.index_list:
            self.port_nav = get_index(self.port_nav,i)
        self.port_nav = self.port_nav.rename(
                columns = {'000905.XSHG':'中证500指数','000300.XSHG':'沪深300指数'},inplace = False)
        return
    
    def get_port_perf(self):
        perf_table = pd.DataFrame()
        for i in self.port_nav.columns[-3:]:
            perf_table = pd.concat([perf_table,get_perf(self.port_nav,i)])
        self.port_perf = perf_table[perf_table.columns[:6]]
        return
    
    def get_ETF_perf(self):
        perf_table = pd.DataFrame()
        for i in self.port_nav.columns[:-3]:
            perf_table = pd.concat([perf_table,get_perf(self.port_nav,i)])
        ETF_perf = perf_table[perf_table.columns[:6]]
        
        index_perf_table = get_ETFindex_perf(self.group,
                                             self.broker_name,self.mkt,
                                             self.last_rpt_period)
        ETF_perf = ETF_perf.reset_index()
        ETF_perf = ETF_perf.rename(columns = {'index':self.mkt+'基金'},inplace = False)
        self.ETF_perf = pd.merge(ETF_perf,index_perf_table,on = [self.mkt+'基金'],how = 'outer')
        self.ETF_perf = self.ETF_perf.set_index([self.mkt+'基金',self.mkt+'代码'])
        return
    
    def get_stk_holding(self):
        self.top_stk,port_ind,self.stk_label = get_holding(self.group,self.broker_name,self.mkt)
        return
    

def get_ETF_reve(ETF_data,time_end):
    ETF_list = list(ETF_data['基金代码'])
    ETF_reve = pd.DataFrame(w.wss(','.join(ETF_list), "return_1w,return_1m,return_ytd","tradeDate="+ time_end.replace('-','') +";annualized=0").Data).T
    ETF_reve.columns = ['近1周收益(%)','近1月收益(%)','年初至今收益(%)']
    ETF_reve.index = ETF_list
    ETF_reve = ETF_reve.reset_index()
    ETF_reve = ETF_reve.rename(columns = {'index':'基金代码'},inplace = False)
    ETF_reve = ETF_reve.merge(ETF_data[['基金代码','基金名称','成立日期','指数名称']],on = '基金代码',how = 'left')
#    del ETF_data
    ETF_reve = ETF_reve.rename(columns = {'指数名称':'标的指数'},inplace = False)
    ETF_reve = ETF_reve.set_index(['基金代码','基金名称'])
    return ETF_reve

def home(time_end,ETF_path):
    '''The home page. '''
    st.title('指数基金行情排序')
    ETF_data = pd.read_excel(ETF_path)
    ETF_reve = get_ETF_reve(ETF_data,time_end)
    
    with st.beta_container():
        col1, col3,col2 = st.beta_columns([1,0.1, 1])
        with col1:
            top10 = ETF_reve.sort_values('近1周收益(%)',ascending = False)[:10]
            st.subheader('近1周业绩排名前10基金')
            for k in range(len(top10)):
                st.info('【'+'】'.join(top10.index[k]))
                st.text('|近1周收益：'+str(np.round(top10['近1周收益(%)'].values[k],2))+ 
                        '%     |近1月收益：'+str(np.round(top10['近1月收益(%)'].values[k],2))+ 
                        '%     |年初至今收益：'+str(np.round(top10['年初至今收益(%)'].values[k],2))+ '%')
                st.text('|追踪指数名称：'+top10['标的指数'].values[k])
            
        with col2:
            top10 = ETF_reve.sort_values('近1周收益(%)',ascending = True)[:10]
            st.subheader('近1周业绩排名后10基金')
            for k in range(len(top10)):
                st.info('【'+'】'.join(top10.index[k]))
                st.text('|近1周收益：'+str(np.round(top10['近1周收益(%)'].values[k],2))+ 
                        '%     |近1月收益：'+str(np.round(top10['近1月收益(%)'].values[k],2))+ 
                        '%     |年初至今收益：'+str(np.round(top10['年初至今收益(%)'].values[k],2))+ '%')
                st.text('|追踪指数名称：'+top10['标的指数'].values[k])
    
    st.text('')
    st.title('指数基金规模分布(亿元)')
    ETF_data.loc[(ETF_data['发行分类'] == '行业')|(ETF_data['发行分类'] == '主题'),'发行分类'] = '行业/主题'
    chart_data = ETF_data[ETF_data['基金类型']!='联接基金'].groupby(['发行分类','基金类型'])['规模合计'].sum()/100000000
    st.text('目前权益类指数基金（不含联接基金）发行规模达到'+str(np.round(chart_data.sum(),2))+
            '亿元。其中规模最大的类别是【行业/主题】型指数基金。')

    del ETF_data
    chart_data = chart_data.reset_index()
    chart_data = np.round(chart_data.pivot_table(index = '发行分类',columns = '基金类型',values = '规模合计'),2).fillna(0)
    chart_data = pd.DataFrame(chart_data.values,index = list(chart_data.index),columns = list(chart_data.columns))
    chart_data = chart_data.sort_values(['ETF'])
    st.bar_chart(chart_data,height = 500,width = 800)
    

def strategy_list_ui(strategy_list_in,strategy_list_out):
    strategy_in = pd.read_excel(strategy_list_in)
    strategy_in = strategy_in.rename(columns = {'Unnamed: 0':'策略名称'},inplace = False)
    strategy_out = pd.read_excel(strategy_list_out)        
    strategy_out = strategy_out.rename(columns = {'Unnamed: 0':'策略名称'},inplace = False)
    strategy_in = strategy_in.sort_values('收益率-1M',ascending = False)
    
    st.title('指数基金策略列表')
    st.text('')
    for k in range(len(strategy_in)):
        with st.beta_container():
            col1, col2,col3 = st.beta_columns([0.6, 1,1])
            with col1:
                st.info(strategy_in['策略名称'].values[k])
                st.text('|'+strategy_in['策略标签'].values[k])
            with col2: 
                st.text('>>>场内基金组合')
                st.text('|收益率-1M:'+strategy_in['收益率-1M'].values[k]+
                        '       |收益率-至今:'+strategy_in['收益率-3M'].values[k])
                st.text('|年化波动率-1M:'+strategy_in['年化波动率-1M'].values[k]+
                        '   |年化波动率-至今:'+strategy_in['年化波动率-3M'].values[k])
            with col3: 
                st.text('>>>场外基金组合')
                st.text('|收益率-1M:'+strategy_out['收益率-1M'].values[k]+
                        '       |收益率-至今:'+strategy_out['收益率-3M'].values[k])
                st.text('|年化波动率-1M:'+strategy_out['年化波动率-1M'].values[k]+
                        '   |年化波动率-至今:'+strategy_out['年化波动率-3M'].values[k])
            
            st.text('——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————')

#                st.write('`{}` rows, `{}` columns'.format(df.shape[0],df.shape[1]))

                
def broker_strategy_ui(broker_strategy_path,strategy_path): 
    strategies = pd.read_excel(strategy_path)
    
    strategy_in = pd.read_excel(broker_strategy_path+'场内策略汇总.xlsx')
    strategy_in = strategy_in.sort_values('收益率-1M',ascending = False)
    strategy_in = strategy_in.rename(columns = {'Unnamed: 0':'策略名称'},inplace = False)
    # Render the header. 
    st.title('指数基金策略组合')
    st.header('2021年度')
    
    # Render file dropbox
    with st.beta_container(): 
        name_select = st.selectbox('', tuple(list(strategy_in['策略名称'])))
        mkt_select = st.selectbox('',('场内','场外'))
        broker_select = strategies.loc[strategies['策略名称'] == name_select,'券商'].values[0]

    with st.beta_container():
        st.subheader(name_select)
        st.info(strategies.loc[strategies['策略名称'] == name_select,'策略要点'].values[0])
        
        df = pd.read_excel(broker_strategy_path+broker_select+mkt_select+'策略结果.xlsx',sheet_name = 'port_perf')
        df = df.rename(columns = {'Unnamed: 0':'策略',
                                  '收益率-2020/12/31至今':'收益率-年初至今',
                                  '年化波动率-2020/12/31至今':'年化波动率-年初至今'},inplace = False)
        df = df.set_index('策略')
        del df['收益率-3M']
        del df['年化波动率-3M']
        
        df_ETF = pd.read_excel(broker_strategy_path+broker_select+mkt_select+'策略结果.xlsx',sheet_name = 'ETF_perf')
        df_conps = [list(z) for z in zip(list(df_ETF[mkt_select+'基金']),list(df_ETF[mkt_select+'权重']))]
        df_conps_plt =  Pie(init_opts = opts.InitOpts(width = '800px',height = '300px')).add(data_pair=df_conps,series_name = mkt_select+'权重',
                           label_opts=opts.LabelOpts(is_show=False, position="right")).set_global_opts(
                                   legend_opts = opts.LegendOpts(type_ = 'scroll',pos_right = '65%',pos_bottom = '100px',orient = 'vertical')).render('pie.html')
        df_ETF = df_ETF.rename(columns = {
                                  '收益率-2020/12/31至今':'收益率-年初至今',
                                  '年化波动率-2020/12/31至今':'年化波动率-年初至今'},inplace = False)
        df_ETF = df_ETF.set_index(mkt_select+'基金',mkt_select+'代码')
        del df_ETF['收益率-3M']
        del df_ETF['年化波动率-3M']
        del df_ETF[mkt_select+'权重']
        
        df_stk = pd.read_excel(broker_strategy_path+broker_select+mkt_select+'策略结果.xlsx',sheet_name = 'top_stk')
        df_stk = df_stk[['股票代码','股票名称','股票权重','区间涨跌幅']]
        df_stk = df_stk.set_index('股票代码')

        df_nav = pd.read_excel(broker_strategy_path+broker_select+mkt_select+'策略结果.xlsx',sheet_name = 'port_nav')
        df_nav['index'] = df_nav['index'].apply(lambda x:str(x)[:10])
        df_nav_plt = Line(init_opts = opts.InitOpts(width = '1100px',height = '500px')).add_xaxis(list(df_nav['index'])
                         ).add_yaxis(name_select,list(round(df_nav[name_select],2))
                         ).add_yaxis('中证500指数',list(round(df_nav['中证500指数'],2))
                         ).add_yaxis('沪深300指数',list(round(df_nav['沪深300指数'],2))
                         ).set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                          ).set_global_opts(yaxis_opts=opts.AxisOpts(min_='dataMin'),datazoom_opts=opts.DataZoomOpts()
                                          ).render('line.html')  
                                          
        df_label = pd.read_excel(broker_strategy_path+broker_select+mkt_select+'策略结果.xlsx',sheet_name = 'top_label')
        df_label['标签占比'] = df_label['标签占比'].apply(lambda x:float(x[:-1]))
        df_label_plt = Bar(init_opts = opts.InitOpts(width = '700px',height = '500px')).add_xaxis(
                list(df_label['一级标签'])).add_yaxis(
                        '标签占比（%）', list(df_label['标签占比'])
                        ).reversal_axis().set_series_opts(label_opts=opts.LabelOpts(position="right")).render("bar.html")    
    
        st.subheader('策略业绩概览')
        with st.beta_container():
            col1, col2 = st.beta_columns([0.8,1])
            with col1:
                st.text('       ')
                st.table(df)
            with col2:
                components.html(open(df_conps_plt, 'r', encoding='utf-8').read(),width = 800,height = 300)
        st.subheader('策略净值表现')
        components.html(open(df_nav_plt, 'r', encoding='utf-8').read(),width = 1100,height = 500)
        st.subheader('所选基金及业绩')
        st.table(df_ETF)
        st.subheader('策略持仓详情')
        with st.beta_container():
            col1, col2 = st.beta_columns([0.8,1])
            with col1:
                st.table(df_stk[:10])
            with col2:
                components.html(open(df_label_plt, 'r', encoding='utf-8').read(),width = 700,height = 500)
 

def construct_strategy_ui(ETF_pool_path):
    with st.beta_container(): 
        st.title('创建指数基金策略组合')
        st.header('基于6大赛道配置指数基金策略')
    
    with st.beta_container():
        mkt_select = st.selectbox('',('场内','场外'))
        ETF = pd.read_excel(ETF_pool_path,sheet_name = mkt_select)    

        col1, col2 = st.beta_columns([1, 1])
        with col1:
            ind1 = st.slider('消费',value = 0.17,min_value = 0.0,max_value = 1.0)
            ind2 = st.slider('科技',value = 0.17,min_value = 0.0,max_value = 1.0)
            ind3 = st.slider('医药',value = 0.17,min_value = 0.0,max_value = 1.0)
            ind4 = st.slider('新能源',value = 0.17,min_value = 0.0,max_value = 1.0)
            ind5 = st.slider('周期',value = 0.16,min_value = 0.0,max_value = 1.0)
            ind6 = st.slider('军工',value = 0.16,min_value = 0.0,max_value = 1.0)

        with col2: 
            fund1 = st.selectbox('消费基金',tuple(set(ETF[ETF['类型']=='消费'][mkt_select+'基金'])))
            fund2 = st.selectbox('科技基金',tuple(set(ETF[ETF['类型']=='科技'][mkt_select+'基金'])))
            fund3 = st.selectbox('医药基金',tuple(set(ETF[ETF['类型']=='周期'][mkt_select+'基金'])))
            fund4 = st.selectbox('新能源基金',tuple(set(ETF[ETF['类型']=='新能源'][mkt_select+'基金'])))
            fund5 = st.selectbox('周期基金',tuple(set(ETF[ETF['类型']=='医药'][mkt_select+'基金'])))
            fund6 = st.selectbox('军工基金',tuple(set(ETF[ETF['类型']=='军工'][mkt_select+'基金'])))
            
        test = pd.DataFrame(columns = ['券商', '策略名称', '策略要点', mkt_select+'基金',mkt_select+'权重','时间'])
        test[mkt_select+'基金'] = [fund1,fund2,fund3,fund4,fund5,fund6]
        ind_sum = ind1+ind2+ind3+ind4+ind5+ind6
        test[mkt_select+'权重'] = [ind1/ind_sum,ind2/ind_sum,ind3/ind_sum,ind4/ind_sum,ind5/ind_sum,ind6/ind_sum]
        test['券商'] = '自主构建'
        test['策略名称'] = '赛道配置策略'
        test['策略要点'] = '基于消费、科技、医药、新能源、周期和军工六个热点赛道，自主定义权重，构建指数基金投资组合。'
        test['时间'] = '2020-12-31'
        test = test.merge(ETF[[mkt_select+'基金',mkt_select+'代码']],on = [mkt_select+'基金'],how = 'left')
        test[mkt_select+'代码'] = test[mkt_select+'代码'].apply(lambda x:str(np.round(x,0)).zfill(8)[:6])
        
        ETF_results = port_ana(group = test,
                               broker_name = '自主构建',
                               mkt = mkt_select,
                               index_list = ['000905.XSHG','000300.XSHG'],
                               last_rpt_period = '20201231')
        ETF_results.get_nav()
        ETF_results.get_port_perf()
        ETF_results.get_ETF_perf()
        ETF_results.get_stk_holding()
    
    with st.beta_container():
        st.info(test['策略要点'].values[0])
        df = ETF_results.port_perf
        df = df.rename(columns = {
                                  '收益率-2020/12/31至今':'收益率-年初至今',
                                  '年化波动率-2020/12/31至今':'年化波动率-年初至今'},inplace = False)
        del df['收益率-3M']
        del df['年化波动率-3M']
        
        df_ETF = ETF_results.ETF_perf.reset_index()
        df_conps = [list(z) for z in zip(list(df_ETF[mkt_select+'基金']),list(df_ETF[mkt_select+'权重']))]
        df_conps_plt =  Pie(init_opts = opts.InitOpts(width = '800px',height = '300px')).add(data_pair=df_conps,series_name = mkt_select+'权重',
                           label_opts=opts.LabelOpts(is_show=False, position="right")).set_global_opts(
                                   legend_opts = opts.LegendOpts(type_ = 'scroll',pos_right = '65%',pos_bottom = '100px',orient = 'vertical')).render('pie.html')
        df_ETF = df_ETF.rename(columns = {
                                  '收益率-2020/12/31至今':'收益率-年初至今',
                                  '年化波动率-2020/12/31至今':'年化波动率-年初至今'},inplace = False)
        df_ETF = df_ETF.set_index(mkt_select+'基金',mkt_select+'代码')
        del df_ETF['收益率-3M']
        del df_ETF['年化波动率-3M']
        del df_ETF[mkt_select+'权重']
        
        df_stk = ETF_results.top_stk
        df_stk = df_stk[['股票代码','股票名称','股票权重','区间涨跌幅']]
        df_stk = df_stk.set_index('股票代码')
        
        df_nav = ETF_results.port_nav.reset_index()
        df_nav['index'] = df_nav['index'].apply(lambda x:str(x)[:10])
        df_nav_plt = Line(init_opts = opts.InitOpts(width = '1100px',height = '500px')).add_xaxis(list(df_nav['index'])
                         ).add_yaxis('赛道配置策略',list(round(df_nav['赛道配置策略'],2))
                         ).add_yaxis('中证500指数',list(round(df_nav['中证500指数'],2))
                         ).add_yaxis('沪深300指数',list(round(df_nav['沪深300指数'],2))
                         ).set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                          ).set_global_opts(yaxis_opts=opts.AxisOpts(min_='dataMin'),datazoom_opts=opts.DataZoomOpts()
                                          ).render('line.html')  
                                          
        df_label = ETF_results.stk_label
        df_label['标签占比'] = df_label['标签占比'].apply(lambda x:float(x[:-1]))
        df_label_plt = Bar(init_opts = opts.InitOpts(width = '700px',height = '500px')).add_xaxis(
                list(df_label['一级标签'])).add_yaxis(
                        '标签占比（%）', list(df_label['标签占比'])
                        ).reversal_axis().set_series_opts(label_opts=opts.LabelOpts(position="right")).render("bar.html")    
    
        st.subheader('策略业绩概览')
        with st.beta_container():
            col1, col2 = st.beta_columns([0.8,1])
            with col1:
                st.text('       ')
                st.table(df)
            with col2:
                components.html(open(df_conps_plt, 'r', encoding='utf-8').read(),width = 800,height = 300)
        st.subheader('策略净值表现')
        components.html(open(df_nav_plt, 'r', encoding='utf-8').read(),width = 1100,height = 500)
        st.subheader('所选基金及业绩')
        st.table(df_ETF)
        st.subheader('策略持仓详情')
        with st.beta_container():
            col1, col2 = st.beta_columns([0.8,1])
            with col1:
                st.table(df_stk[:10])
            with col2:
                components.html(open(df_label_plt, 'r', encoding='utf-8').read(),width = 700,height = 500)


def label_strategy_ui(ETF_label_path):
    ETF_label = pd.read_excel(ETF_label_path)
    ETF_label = ETF_label[ETF_label['股票权重']>15]
    mkt_select = st.selectbox('',('场内','场外'))
    if mkt_select == '场内':
        ETF_label = ETF_label[ETF_label['基金类型']=='ETF']
    else:
        ETF_label = ETF_label[ETF_label['基金类型']!='ETF']


    def pretty(s: str) -> str:
        try:
            return dict(js="JavaScript")[s]
        except KeyError:
            return s.capitalize()
    selection = st.multiselect("热门标签", options=list(set(ETF_label['一级标签'])), 
                               default=['太阳能光伏','医疗服务','证券'], format_func=pretty)
    weight_list = []
    weight_sum = 0
    fund_list = []
    code_list = []
    for l in selection:
        col1, col2 = st.beta_columns([1, 1])
        with col1:
            ind_l = st.slider(l,value = 0.17,min_value = 0.0,max_value = 1.0)
            weight_list = weight_list + [ind_l]
            weight_sum = weight_sum+ind_l
        with col2: 
            ETF_l = ETF_label[ETF_label['一级标签']==l].sort_values('股票权重',ascending = False)
            fund_l = st.selectbox(l,tuple(set(ETF_l['基金名称'])))
            code_l = ETF_l[ETF_l['基金名称']==fund_l]['基金代码'].values[0]
            fund_list = fund_list + [fund_l]
            code_list = code_list + [code_l]
            
    test = pd.DataFrame(columns = ['券商', '策略名称', '策略要点', mkt_select+'基金',mkt_select+'权重','时间'])
    test[mkt_select+'基金'] = fund_list
    test[mkt_select+'代码'] = code_list
    test[mkt_select+'权重'] = [i/weight_sum for i in weight_list]
    test['券商'] = '自主构建'
    test['策略名称'] = '标签配置策略'
    test['策略要点'] = '基于热门标签族谱，自主定义权重，构建指数基金投资组合。'
    test['时间'] = '2020-12-31'
    test[mkt_select+'代码'] = test[mkt_select+'代码'].apply(lambda x:x[:6])
    
    ETF_results = port_ana(group = test,
                           broker_name = '自主构建',
                           mkt = mkt_select,
                           index_list = ['000905.XSHG','000300.XSHG'],
                           last_rpt_period = '20201231')
    ETF_results.get_nav()
    ETF_results.get_port_perf()
    ETF_results.get_ETF_perf()
    ETF_results.get_stk_holding()

    with st.beta_container():
        st.info(test['策略要点'].values[0])
        df = ETF_results.port_perf
        df = df.rename(columns = {
                                  '收益率-2020/12/31至今':'收益率-年初至今',
                                  '年化波动率-2020/12/31至今':'年化波动率-年初至今'},inplace = False)
        del df['收益率-3M']
        del df['年化波动率-3M']
        
        df_ETF = ETF_results.ETF_perf.reset_index()
        df_conps = [list(z) for z in zip(list(df_ETF[mkt_select+'基金']),list(df_ETF[mkt_select+'权重']))]
        df_conps_plt =  Pie(init_opts = opts.InitOpts(width = '800px',height = '300px')).add(data_pair=df_conps,series_name = mkt_select+'权重',
                           label_opts=opts.LabelOpts(is_show=False, position="right")).set_global_opts(
                                   legend_opts = opts.LegendOpts(type_ = 'scroll',pos_right = '65%',pos_bottom = '100px',orient = 'vertical')).render('pie.html')
        df_ETF = df_ETF.rename(columns = {
                                  '收益率-2020/12/31至今':'收益率-年初至今',
                                  '年化波动率-2020/12/31至今':'年化波动率-年初至今'},inplace = False)
        df_ETF = df_ETF.set_index(mkt_select+'基金',mkt_select+'代码')
        del df_ETF['收益率-3M']
        del df_ETF['年化波动率-3M']
        del df_ETF[mkt_select+'权重']
        
        df_stk = ETF_results.top_stk
        df_stk = df_stk[['股票代码','股票名称','股票权重','区间涨跌幅']]
        df_stk = df_stk.set_index('股票代码')
        
        df_nav = ETF_results.port_nav.reset_index()
        df_nav['index'] = df_nav['index'].apply(lambda x:str(x)[:10])
        df_nav_plt = Line(init_opts = opts.InitOpts(width = '1100px',height = '500px')).add_xaxis(list(df_nav['index'])
                         ).add_yaxis('标签配置策略',list(round(df_nav['标签配置策略'],2))
                         ).add_yaxis('中证500指数',list(round(df_nav['中证500指数'],2))
                         ).add_yaxis('沪深300指数',list(round(df_nav['沪深300指数'],2))
                         ).set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                          ).set_global_opts(yaxis_opts=opts.AxisOpts(min_='dataMin'),datazoom_opts=opts.DataZoomOpts()
                                          ).render('line.html')  
                                          
        df_label = ETF_results.stk_label
        df_label['标签占比'] = df_label['标签占比'].apply(lambda x:float(x[:-1]))
        df_label_plt = Bar(init_opts = opts.InitOpts(width = '700px',height = '500px')).add_xaxis(
                list(df_label['一级标签'])).add_yaxis(
                        '标签占比（%）', list(df_label['标签占比'])
                        ).reversal_axis().set_series_opts(label_opts=opts.LabelOpts(position="right")).render("bar.html")    
    
        st.subheader('策略业绩概览')
        with st.beta_container():
            col1, col2 = st.beta_columns([0.8,1])
            with col1:
                st.text('       ')
                st.table(df)
            with col2:
                components.html(open(df_conps_plt, 'r', encoding='utf-8').read(),width = 800,height = 300)
        st.subheader('策略净值表现')
        components.html(open(df_nav_plt, 'r', encoding='utf-8').read(),width = 1100,height = 500)
        st.subheader('所选基金及业绩')
        st.table(df_ETF)
        st.subheader('策略持仓详情')
        with st.beta_container():
            col1, col2 = st.beta_columns([0.8,1])
            with col1:
                st.table(df_stk[:10])
            with col2:
                components.html(open(df_label_plt, 'r', encoding='utf-8').read(),width = 700,height = 500)

    
def main():
    '''Add control flows to organize the UI sections. '''
#    st.sidebar.image('./docs/logo.png', width=250)
    st.sidebar.write('') # Line break
    st.sidebar.header('被动指数投研系统')
    side_menu_selectbox = st.sidebar.radio(
        '菜单', ('指数行情与业绩概览', '策略列表', '策略详情','我的策略'))
    if side_menu_selectbox == '指数行情与业绩概览':
        home(time_end = datetime.now().strftime('%Y-%m-%d'),
             ETF_path= 'D:\\K\\project_indexfund\\202106\\data\\权益类ETF产品列表.xlsx')
    elif side_menu_selectbox == '策略列表':
        strategy_list_ui(strategy_list_in = 'D:\\K\\project_indexfund\\202106\\data\\strategies\\场内策略汇总.xlsx',
                         strategy_list_out = 'D:\\K\\project_indexfund\\202106\\data\\strategies\\场外策略汇总.xlsx',)
    elif side_menu_selectbox == '策略详情':
        side_menu_sub_selectbox = st.sidebar.radio('指数基金策略组合', ('按赛道配置基金', '推荐策略配置基金', '标签族谱配置基金'))
        if side_menu_sub_selectbox == '按赛道配置基金':
            construct_strategy_ui(ETF_pool_path = 'D:\\K\\project_indexfund\\202106\\data\\ind_ETFs\\ETF_pool.xlsx')
            agree = st.checkbox('一键下单')
            if agree:
                st.write('交易成功!')
        elif side_menu_sub_selectbox == '推荐策略配置基金':
            broker_strategy_ui(broker_strategy_path = 'D:\\K\\project_indexfund\\202106\\data\\strategies\\',
                               strategy_path = 'D:\\K\\project_indexfund\\202106\\data\\ETF_strategies.xlsx')
            agree = st.checkbox('一键下单')
            if agree:
                st.write('交易成功!')
        elif side_menu_sub_selectbox == '标签族谱配置基金':
            label_strategy_ui(ETF_label_path = 'D:\\K\\project_indexfund\\202106\\data\\ETF基金_标签汇总.xlsx')
            agree = st.checkbox('一键下单')
            if agree:
                st.write('交易成功!')

    elif side_menu_selectbox == '我的策略':
        st.text('')        
        if st.button('一键下单'):
            st.write('交易成功！')
        else:
            st.write('等待交易')

if __name__ == '__main__': 
    st.set_page_config(page_title='指数基金量化投资组合管理', layout='wide', initial_sidebar_state='auto')
    try: 
        main()
    except: 
        st.error('Oops! Something went wrong...Please check your input.\nIf you think there is a bug, please open up an [issue](https://github.com/luxin-tian/mosco_ab_test/issues) and help us improve. ')
        raise
