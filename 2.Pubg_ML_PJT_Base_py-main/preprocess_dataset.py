import numpy as np
import pandas as pd

def preprocess(df):
    df = __delete_nan_data(df)
    new_col_name = "match_types"
    df[new_col_name] = __convert_match_type_column(df,"matchType")
    df = __change_nan_points(df)
    df = __one_hot_encode_data_frame(df, new_col_name)
    df = __select_features(df)
    return df

  
def __delete_nan_data(df):
    return df.dropna()

  
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


def __select_features(df):
    main_columns = ["winPlacePerc", "walkDistance", "boosts", "weaponsAcquired"]
    kill_columns = ["kills", "damageDealt"]
    match_type_columns = df.columns[df.columns.str.contains("match_types")]
    deleted_columns = list(set(df.columns)-set(main_columns)-set(kill_columns)-set(match_type_columns))
    # 전체칼럼 - 메인칼럼 - 칼럼 - 매치타입칼럼 
    return df.drop(columns=deleted_columns)


