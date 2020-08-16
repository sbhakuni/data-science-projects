# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:01:04 2019

@author: bhaku
"""
""" here i have done data cleaning and made an modular script for later analysis"""

import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
from dateutil.parser import parse
import time
from datetime import timedelta
import pickle
import os
import pandas_profiling as pf
from IPython.display import display, HTML
import matplotlib.pyplot as plt

###############################################################################
"""Take file from TRTH data and clean it, index on datetime timestamp and save each stock data 
into its own CSV file"""

def CleanTRTHdata(data, wd = None):
    #data : csv.gz file with stocks from Thomson Rueters
    #wd: working directory to save file into if specified
    
    #drop time zone and domain column 
    data = data.drop(['Domain', 'GMT Offset'], axis = 1)


    #split the data frame into a dictionary containing smaller dataframes 
    #comprised of each stock
    data_sep = {}

    for key, value in data.groupby(data['#RIC']):
        data_sep[key] = value
 
    #Index each dataframe in dictionary on their date time
    for i, ticker in enumerate(data_sep):
        data_sep[ticker]['Date-Time'] = data_sep[ticker]['Date-Time'].str.replace('T', ' ' )
        data_sep[ticker]['Date-Time'] = data_sep[ticker]['Date-Time'].str.replace('Z', '' )
        data_sep[ticker]['Date-Time'] = pd.to_datetime(data_sep[ticker]['Date-Time'])
        data_sep[ticker]['Date-Time'] = data_sep[ticker]['Date-Time'] - pd.DateOffset(hours = 4)
        data_sep[ticker].set_index('Date-Time', inplace = True)
        
    if wd is None:
        path = os.getcwd()
    else:
        path = os.chdir(wd)
    for key in data_sep:
        new_key =  key.split('.')[0]
        df = data_sep[key]
        df.to_csv(path + '/{}.csv'.format(new_key))
    
    
    

###################################################    
"""Saving each stock in dataframe to its own csv"""
 
def SaveStockData(data_dict, wd = None):
    #data_dict: dictionary of dataframes where each dataframe is each stock
    #wd: working directory to save file into if specified
    if wd is None:
        path = os.getcwd()
    else:
        path = os.chdir(wd)
    for key in data_dict:
        new_key =  key.split('.')[0]
        df = data_dict[key]
        df.to_csv(path + '/{}.csv'.format(new_key))
    
    
    

###############################################################################
"""Slice dataframe to only have the orders from the hours the exchange is open"""
    
    
def TradingHours(df):
    #df : dataframe indexed by timestamp that will be sliced
    new_df = pd.DataFrame()
    

    #number of days in dataframe
    trading_days = (df.index[-1] - df.index[0]).days + 1
    
    #get the trading hour rows for each day
    for i in range(trading_days):
        day = str(str(df.index[0].year) + '-' + str(df.index[0].month) + '-' + str(df.index[0].day + i))
        
        #early close for july 3rd
        if day == "2019-7-3":
            start_time = (day + ' ' + '09:30:00')
            end_time = (day + ' ' + '12:59:59')
        else:
            start_time = (day + ' ' + '09:30:00')
            end_time = (day + ' ' + '15:59:59')
            
        df_day = df.loc[start_time : end_time]
        
        if df_day.empty:
            i = i+1
        else:
            if new_df.empty:
                new_df = df_day
            else:
                new_df = pd.concat([new_df, df_day], axis = 0)
            
        
    return new_df

        

###############################################################################
"""Load the csv file indexed on datetime"""

def LoadCSV(file_name, Trading_hours = True):
    #file: file path with csv is located on computer
    
    f = pd.read_csv(file_name, index_col = 'Date-Time')
    f.index = pd.to_datetime(f.index)
    
    if Trading_hours == True:
        f = TradingHours(f)
    
    
    return f



###############################################################################
"""Aggregation function to create dictionary of smaller dataframes based on time interval"""



def Aggregation(df, windowsize_seconds = 60):
    #df : dataframe of stock during trading hours to slice into aggregation windows
    #windowsize_seconds = size of window to split the data
    
    agg_dict = {}
    
    
    #number of trading days
    trading_days = (df.index[-1] - df.index[0]).days + 1
    
    
    for i in range(trading_days):
        day = day = str(str(df.index[0].year) + '-' + str(df.index[0].month) + '-' + str(df.index[0].day + i))
        daily_df = df.loc[day]
        #number of windows in a trading day
        if day == '2019-7-3':
            num_windows = int(12600/windowsize_seconds)
        else:    
            num_windows = int(23400/windowsize_seconds)
        if daily_df.empty:
            i= i + 1
        else:
            start_time = daily_df.index[0].replace(second=0, microsecond = 0, nanosecond = 0)
            for i in range(num_windows):
                end_time = start_time + timedelta(seconds= windowsize_seconds)
                intra_df = daily_df[start_time:end_time]
                agg_dict[end_time] = intra_df
                #print(intra_df.head())
                start_time = end_time
                
                
    return agg_dict
                

"""LIQUIDITY MEASURE FUNCTIONS"""
###############################################################################
"""Total number of shares traded in a time interval"""


def TradingVolume(df):
    #df : dataframe of stock from TRTH 
    
    if df.empty:
        volume = 0
    else:
        volume = df['Volume'].sum()
    return volume

    

###############################################################################
"""Total number of dollar value traded in a time interval"""

    
def Turnover(df):
    #df : dataframe of stock from TRTH 
    
    df = df[df['Type'] == 'Trade']
    if df.empty:
        turnover = 0
    else:
        turnover = (df['Price']*df['Volume']).sum()
    return turnover



###############################################################################
"""The sum of bid and ask volume in a time interval"""    


def Depth(df):
    #df : dataframe of stock from TRTH 
    
    if df.empty:
        depth = 0
    else:  
        depth = df['Ask Size'].sum() + df['Bid Size'].sum()
    return depth



###############################################################################
"""Log of depth in an attempt to imporve distribution properties"""


def LogDepth(df):
    #df : dataframe of stock from TRTH 
    
    if df.empty:
        log_depth = 0
    else:
        log_ask = np.log(df['Ask Size'])
        log_bid = np.log(df['Bid Size'])
        log_depth = log_ask.sum() + log_bid.sum()
    return log_depth


    
###############################################################################
"""Average of quoted bid and ask depths in dollar terms"""

def DollarDepth(df):
    #df : dataframe of stock from TRTH 
    
    df_quote = df[df['Type'] == 'Quote']
    if df_quote.empty:
        d_depth = None
    else:
        p_ask = df_quote['Ask Price'][-1]
        p_bid = df_quote['Bid Price'][-1]
    
        d_depth = (df_quote['Ask Size'].sum()*p_ask + df_quote['Bid Size'].sum()*p_bid)/2
    return d_depth
    
    
    
###############################################################################
"""Number of transactions in a time interval"""

def TransactionCount(df):
    #df : dataframe of stock from TRTH 
    
    df = df[df['Type'] == 'Trade']    
    return df.shape[0]



###############################################################################
"""Dollar spread at time t"""

def AbsoluteSpread(df):
    #df : dataframe of stock from TRTH 
       
    df_quote = df[df['Type'] == 'Quote']
    if df_quote.empty:
        spread = None
    else:
        spread = df_quote['Ask Price'][-1] - df_quote['Bid Price'][-1]
    return spread



###############################################################################
"""Log of absolute spread in attempt to improve distirbution properties"""

def LogSpread(df):
    #df : dataframe of stock from TRTH 
    
    abs_spread = AbsoluteSpread(df)
    if abs_spread is None:
        log_spread = None
    else:
        log_spread = np.log(abs_spread)
    return log_spread



###############################################################################
"""Proportional spread calculated with mid price"""

def RelativeSpreadMidPrice(df):
    #df : dataframe of stock from TRTH 
    
    df_quote = df[df['Type'] == 'Quote']
    if df_quote.empty:
        spread = None
    else:
        spread = (2*(df_quote['Ask Price'][-1] - df_quote['Bid Price'][-1]))/(df_quote['Ask Price'][-1] + df_quote['Bid Price'][-1])
    
    return spread



###############################################################################
"""Spread divided by last paid price"""

def RelativeSpreadLastPrice(df):
    #df : dataframe of stock from TRTH 
    
    df_trade = df[df['Type'] == 'Trade']
    df_quote = df[df['Type'] == 'Quote']
    if (df_trade.empty or df_quote.empty):
        spread = None
    else:
        spread = (df_quote['Ask Price'][-1] - df_quote['Bid Price'][-1]) / df_trade['Price'][-1]
    
    return spread



###############################################################################
"""Log Spread of Bid and Ask Price"""

def RelativeSpreadLogPrice(df):
    #df : dataframe of stock from TRTH 
    
    df_quote = df[df['Type'] == 'Quote']
    if df_quote.empty:
        spread = None
    else:
        spread = np.log((df['Ask Price'][-1])/df['Bid Price'][-1])
    return spread



###############################################################################
"""Log  of the Relative Log Spread for distirubtion properties"""

def LogRelativeSpreadLogPrice(df):
    #df : dataframe of stock from TRTH 
    
    rel_log = RelativeSpreadLogPrice(df)
    if rel_log is None:
        spread = None
    else:
        spread = np.log(rel_log)
    return spread



###############################################################################
"""Trade price minus mid price; is Effective spread is smaller than half of absolute spread
this reflects trading inside the quotes"""

def EffectiveSpread(df):
    #df : dataframe of stock from TRTH 
    
    df_quote = df[df['Type'] == 'Quote']
    df_trade = df[df['Type'] == 'Trade']
    if (df_quote.empty or df_trade.empty):
        spread = None
    else:
        mid_price = (df_quote['Ask Price'][-1] + df_quote['Bid Price'][-1])/2
        spread = np.abs(df_trade['Price'][-1] - mid_price)
    
    return spread
    


###############################################################################
"""Divding effective spread by last trade price"""

def RelativeEffectiveSpreadLastPrice(df):
    #df : dataframe of stock from TRTH 
    
    df_trade = df[df['Type'] == 'Trade']
    eff_spread = EffectiveSpread(df)
    if (df_trade.empty or eff_spread is None):
        spread = None
    else:
        spread = eff_spread/df_trade['Price'][-1]
    
    return spread



###############################################################################
"""Divide Effective spread by mid price of quotes"""

def RelativeEffectiveSpreadMidPrice(df):
    #df : dataframe of stock from TRTH
    
    df_quote = df[df['Type'] == 'Quote']
    eff_spread = EffectiveSpread(df)
    if (df_quote.empty or eff_spread is None):
        spread = None
    else:
        mid_price = (df_quote['Ask Price'][-1] + df_quote['Bid Price'][-1])/2
        spread = eff_spread/mid_price
    
    return spread
    
    

###############################################################################
"""Absolute spread divided by the Log Depth"""

def QuoteSlope(df):
    #df : dataframe of stock from TRTH
    
    abs_spread = AbsoluteSpread(df)
    d_log = LogDepth(df)
    if(abs_spread is None or d_log is None):
        slope = None
    else:
        slope = abs_spread/d_log
    
    return slope



###############################################################################
"""Log Relative spread divided by the log depth"""

def LogQuoteSlope(df):
    #df : dataframe of stock from TRTH
    
    rel_spread = RelativeSpreadLogPrice(df)
    d_log = LogDepth(df)
    if(rel_spread is None or d_log is None):
        slope = None
    else:
        slope = rel_spread/d_log
    
    return slope



###############################################################################
"""Adjusting the Log slope by a correction term"""

def AdjustedLogQuoteSlope(df):
    #df : dataframe of stock from TRTH
    
    log_qs = LogQuoteSlope(df)
    df_quote = df[df['Type'] == 'Quote']
    if(log_qs is None or df_quote.empty):
        adj_slope = None
    else:
        ask_vol = df_quote['Ask Size'].sum()
        bid_vol = df_quote['Bid Size'].sum()
        correction = (1 + np.abs(np.log(ask_vol/bid_vol)))
        adj_slope = log_qs*correction
    
    return adj_slope



###############################################################################
"""Relative spread midprice divided by dollar depth"""

def CompositeLiquidity(df):
    #df : dataframe of stock from TRTH
    
    rel_midp = RelativeSpreadMidPrice(df)
    d_depth = DollarDepth(df)
    if(rel_midp is None or d_depth is None):
        comp_liq = None
    else:
        comp_liq = rel_midp/d_depth
    
    return comp_liq



###############################################################################
"""Turnover divided by the rate of return for the period"""

def LiquidityRatio1(df):
    #df : dataframe of stock from TRTH
    
    df_trade = df[df['Type'] == 'Trade']
    turnover = Turnover(df)
    if(df_trade.empty):
        ratio = None
    else:
        return_rate = np.abs((df_trade['Price'][-1] - df_trade['Price'][0])/df_trade['Price'][0])
        if return_rate == 0:
            ratio = 0
        else:
            ratio = turnover/return_rate
    
    return ratio




###############################################################################
"""Number of transactions times the turnover"""

def FlowRatio(df):
    #df : dataframe of stock from TRTH
    
    nt = TransactionCount(df)
    vt = Turnover(df)
    ratio = nt*vt
    
    return ratio

    

###############################################################################
"""Compares depth measured as market imbalance to turnover"""

def OrderRatio(df):
    #df : dataframe of stock from TRTH
    
    df_quote = df[df['Type'] == 'Quote']
    if df_quote.empty:
        ratio = None
    else:
        vt = Turnover(df)
        if vt == 0:
            ratio = 0
        else:
            ratio = np.abs(df_quote['Bid Size'].sum() - df_quote['Ask Size'].sum())/vt
    
    return ratio
    
###############################################################################
""" Aggregated price"""    
def Price(df):
    #df : dataframe of stock from TRTH 
    df_quote = df[df['Type'] == 'Quote']
    df_trade = df[df['Type'] == 'Trade']
    if (df_quote.empty or df_trade.empty):
        price = None
    else:
        price=df_trade['Price'][-1]
    
    return price 



###############################################################################
"""Function to apply measure on Aggregated Data and Return Series of quantitative values"""

def ApplyMeasure(agg_dict, measure):
    #agg_dict: dictionary of scliced dataframes
    #measure: liqudity measure to run on dataframes
    #series_name: name for column of values in your series
    
    #empty dictionary to store values with corresponding timestamp
    measure_value = {}
    
    for index, timestamp in enumerate(agg_dict):
        value = measure(agg_dict[timestamp])
        measure_value[timestamp] = value
    
    
    #convert measure dictionary into Pandas Series
    df = pd.DataFrame.from_dict(measure_value, orient = 'index', columns= [measure.__name__])
    
    
    return df





###############################################################################
"""Function to apply all measures on an aggregated dictionary and save them in a data frame"""

def ApplyAllMeasures(df, windowsize_seconds = 60):
    #df: dataframe of stock data to aggregate
    #windowsize_seconds: how long the time interval for slicing the data frame  
    
    #slice the dataframe into smaller ones of chosen time size
    agg_dict = Aggregation(df, windowsize_seconds)
    
    dataframe = pd.DataFrame()
    all_measures = [TradingVolume, Turnover, Depth, LogDepth, DollarDepth, TransactionCount,
                    AbsoluteSpread, LogSpread, RelativeSpreadMidPrice, RelativeSpreadLastPrice, 
                    RelativeSpreadLogPrice, LogRelativeSpreadLogPrice, EffectiveSpread, 
                    RelativeEffectiveSpreadLastPrice, RelativeEffectiveSpreadMidPrice,
                    QuoteSlope, LogQuoteSlope, AdjustedLogQuoteSlope, CompositeLiquidity,
                    LiquidityRatio1, FlowRatio, OrderRatio, Price]
    
    for i in range(len(all_measures)):
        df = ApplyMeasure(agg_dict, all_measures[i])
        
        if dataframe.empty:
            dataframe = df
        else:
            dataframe = pd.concat([dataframe, df], axis = 1)
    
    return dataframe

###############################################################################
"""Function to put a price jump marker of the rows with the corressponding jump"""
def jump(df, percentage_jump=1):
    df['jump'] = (df.Price.diff()/df['Price'])*100
    df['jump']=np.absolute(df['jump'].fillna(0))
    df['jump_event_marker'] = np.where(df['jump']>=percentage_jump, 'yes', 'no')
    return df


###############################################################################
"""functions to map timestamps(jump) from low freq to high freq"""
def jumpMarkerMatcher(df_low_freq,df_high_freq):

    A=df_low_freq.loc[df_low_freq['jump_event_marker']=='yes']
    df_high_freq['jump_marker']=np.where(df_high_freq.index.isin(A.index),'yes','no')
    return df_high_freq