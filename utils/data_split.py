from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def data_split(split_type, train_df, target_column):
    """
        funstion_name : data_split
        use : 원하는 type에 맞춰 data split을 하기위한 함수
        INPUT : 
            split_type[String] : 'random/time/randomcv'
            train_df[pd.DataFrame] : train dataframe
            target_column : 정답을 맞출 target_column
        RETURN :
            x_train : x train data
            x_valid : x valid data
            y_train : train target data
            y_valid : valid target data
    """
    if split_type == "random":
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_df, 
            train_df[target_column], 
            test_size=0.2,
            random_state=42,
            stratify=train_df[target_column]
        )
    elif split_type == "time":
        holdout_start = 202307
        holdout_end = 202312

        x_train = train_df[~((train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end))].drop(columns=[target_column])
        y_train = train_df[~((train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end))][target_column]

        x_valid = train_df[(train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end)].drop(columns=[target_column])
        y_valid = train_df[(train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end)][target_column]
    # elif split_type == "randomcv":
    #     # cross validation
    #     x_train = train_df.drop(drop_colunm, axis = 1)
    #     y_train = train_df[target_colunm].astype(int)
    #     x_train_list = []
    #     y_train_list = []
    #     x_valid_list = []
    #     y_valid_list = []
    #     skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        
    #     for train_index, valid_index in skf.split(x_train, y_train):
    #         x_train_0, x_valid_0 = x_train.iloc[train_index], x_train.iloc[valid_index]
    #         x_train_list.append(x_train_0)
    #         x_valid_list.append(x_valid_0)
    #         y_train_0, y_valid_0 = y_train.iloc[train_index], y_train.iloc[valid_index]
    #         y_train_list.append(y_train_0)
    #         y_valid_list.append(y_valid_0)
    #     return x_train_list, x_valid_list, y_train_list, y_valid_list    
    else:
        raise Exception("Invalid model_name. (data_split.py)")
        
    return x_train, x_valid, y_train, y_valid