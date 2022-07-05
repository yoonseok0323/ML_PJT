import numpy as np
import pandas as pd

def preprocess(df):
    df = __delete_nan_data(df)
    #df = __concat_data(df)
    new_col_name = "match_types"
    df[new_col_name] = __convert_match_type_column(df,"matchType")
    df = __change_nan_points(df)
    df = __one_hot_encode_data_frame(df, new_col_name)
    df = __select_features(df)
    #__vif(df)
    return df

  
def __delete_nan_data(df):
    #print(df['kills'])
    return df.dropna()

def __concat_data(df):
    df['kapoints']= df['assists']+df['kills']
    df['useditems']= df['boosts']+df['heals']
    df['totaldistance'] = df['walkDistance'] + df['swimDistance']+df['rideDistance']
    #print(df['totaldistance'])
    return df

def __convert_match_type_column(prepro_df,encoding_feature):
    encoded = prepro_df[encoding_feature].agg(preprocessing_match_type)
    return encoded

  
def preprocessing_match_type(match_type):
    standard_matches = ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"]
    if match_type in standard_matches:
        return match_type
    else:
        return "others" 

      
def __change_nan_points(df):
    kill_rank_win_points = ["killPoints", "rankPoints", "winPoints"]
    match_types_list = list(df.match_types.unique())
    for col in kill_rank_win_points:
        if col != "rankPoints":
            cond0 = df[col] == 0
            cond1 = df[col] != 0
        else:
            cond0 = df[col] == -1
            cond1 = df[col] != -1
        for m_type in match_types_list:
            cond2 = df.match_types == m_type
            mean = df[cond1 & cond2][col].mean()
            std = df[cond1 & cond2][col].std()
            size = df[cond0 & cond2][col].count()
            if m_type != 'others' or col == "rankPoints":
                rand_points = np.random.randint(mean-std, mean+std, size=size)
            else:
                rand_points = np.array([mean]*size)
            df[col].loc[cond0 & cond2] = rand_points
    return df

  
def __one_hot_encode_data_frame(df, encoding_feature):
    df = pd.get_dummies(df, columns=[encoding_feature])
    return df

def __vif(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    variance_inflation_factor(df.values, 0)
    pd.DataFrame({
    "VIF Factor": [variance_inflation_factor(df.values, idx) for idx in range(df.shape[1])],
    "features": df.columns})

def __select_features(df):
    #main_columns = ["winPlacePerc",'walkDistance', 'kills','boosts','heals', 'killStreaks', 'longestKill','rideDistance']
    #kill_columns = ["kills",'damageDealt','assists','headshotKills','killStreaks','longestKill']
    #sub_columns = ['weaponsAcquired', 'damageDealt', 'headshotKills','assists',  'DBNOs']
    #point_columns = ['killPoints','rankPoints','winPoints']
    # match_type_columns = df.columns[df.columns.str.contains("match_types")]
    #deleted_columns = list(set(df.columns)-set(main_columns)-set(sub_columns)-set(point_columns)-set(match_type_columns))
    match_type_columns =['match_types_squad', 'match_types_duo-fpp','match_types_duo']
    
    #deleted_columns = list(set(df.columns)-set(main_columns)-set(match_type_columns))
        #'killStreaks','longestKill'
    # all_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs',
    #     'heals',  'killPoints', 'kills',
    #    'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
    #    'numGroups', 'rankPoints', 'revives', 'rideDistance',
    #    'swimDistance',  'walkDistance',
    #    'weaponsAcquired', 'winPoints', 'winPlacePerc']
    
    lasso_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs'
                     ,'heals',
                     'killPoints',
                     'kills',
                     'killStreaks',
                     'longestKill',
                     'matchDuration',
                     'maxPlace',
                     'numGroups',
                     'rankPoints',
                     'revives',
                     'rideDistance',
                     'swimDistance',
                     'walkDistance',
                     'weaponsAcquired',
                     'winPoints','winPlacePerc'
                     ]

    deleted_columns = list(set(df.columns)-set(lasso_columns)-set(match_type_columns))
    
    return df.drop(columns=deleted_columns)
    
    # totaldistance', 'kapoints','useditems'
    #deleted_columns = list(set(df.columns)-set(main_columns)-set(kill_columns)-set(point_columns)-set(match_type_columns))
    #kill_columns = ["kills", "damageDealt",'assists','killPlace','headshotKills','killStreaks','longestKill']
    #point_columns = ['killPoints','rankPoints','winPoints']
    
    # df = df.columns[df.columns.str.contains("match_types")]
    # return df

