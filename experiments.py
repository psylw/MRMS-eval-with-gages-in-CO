# experiments.py

import pandas as pd

def original(df,state):
    return df,state

def pre_2021(df,state):
    df = df.copy()
    df['year'] = df['start'].dt.year
    df = df[df['year'] < 2021]

    state = state[state.year<2021]

    return df.drop(columns='year'),state

def post_2021(df,state):
    df = df.copy()
    df['year'] = df['start'].dt.year
    df = df[df['year'] >= 2021]

    state = state[state.year>=2021]

    return df.drop(columns='year'),state

def jja(df,state):

    return df[df['month'].isin([6, 7, 8])],state[state['month'].isin([6, 7, 8])]

def pre_2021_jja(df,state):
    df,state = pre_2021(df,state)

    return df[df['month'].isin([6, 7, 8])],state[state['month'].isin([6, 7, 8])]

def post_2021_jja(df,state):
    df,state = post_2021(df,state)
    return df[df['month'].isin([6, 7, 8])],state[state['month'].isin([6, 7, 8])]

def add_year_as_feature(df,state):
    df = df.copy()
    df['year'] = df['start'].dt.year

    return df,state

def nr_rmse(df,state):
    df = df.copy()
    df['norm_diff'] = df['norm_diff'] / df['max_mrms']

    return df,state

def nr_rmse_pre(df,state):
    df = df.copy()
    df['norm_diff'] = df['norm_diff'] / df['max_mrms']

    return pre_2021(df,state)

def nr_rmse_post(df,state):
    df = df.copy()
    df['norm_diff'] = df['norm_diff'] / df['max_mrms']

    return post_2021(df,state)

def mean_error(df,state):
    df = df.copy()
    df = df[df['total_mrms_accum'] > 1].reset_index(drop=True)
    df = df.dropna()
    df['norm_diff'] = pd.read_feather('output/experiments/mean_error_values')
    return df,state

def mean_error_pre(df,state):
    df = df.copy()
    df = df[df['total_mrms_accum'] > 1].reset_index(drop=True)
    df = df.dropna()
    df['norm_diff'] = pd.read_feather('output/experiments/mean_error_values')
    return pre_2021(df,state)

def mean_error_post(df,state):
    df = df.copy()
    df = df[df['total_mrms_accum'] > 1].reset_index(drop=True)
    df = df.dropna()
    df['norm_diff'] = pd.read_feather('output/experiments/mean_error_values')
    return post_2021(df,state)

def mean_error_pre_2021_jja(df,state):
    df = df.copy()
    df = df[df['total_mrms_accum'] > 1].reset_index(drop=True)
    df = df.dropna()
    df['norm_diff'] = pd.read_feather('output/experiments/mean_error_values')

    return pre_2021_jja(df,state)

def mean_error_post_2021_jja(df,state):
    df = df.copy()
    df = df[df['total_mrms_accum'] > 1].reset_index(drop=True)
    df = df.dropna()
    df['norm_diff'] = pd.read_feather('output/experiments/mean_error_values')
    return post_2021_jja(df,state)

def mean_error_year(df,state):
    df = df.copy()
    df = df[df['total_mrms_accum'] > 1].reset_index(drop=True)
    df = df.dropna()
    df['norm_diff'] = pd.read_feather('output/experiments/mean_error_values')
    return add_year_as_feature(df,state)
