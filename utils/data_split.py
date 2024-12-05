from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def data_split(split_type, train_df, target_column):
    """
    데이터셋을 지정된 방식에 따라 학습 및 검증 세트로 분할.

    Args:
        split_type (str): 데이터 분할 방식, "random" 또는 "time".
        train_df (pd.DataFrame): 분할할 데이터 프레임.
        target_column (str): 예측 대상이 되는 타겟 컬럼 이름.

    Raises:
        Exception: split_type이 "random" 또는 "time"이 아닐 경우 예외 발생.

    Returns:
        tuple: 학습 및 검증 세트로 분할된 입력과 타겟 (x_train, x_valid, y_train, y_valid).
    
    random:
        - 데이터를 무작위로 분할.
        - train_test_split을 사용하여 데이터의 80%를 학습, 20%를 검증으로 분할.
        - target_column 기준으로 층화(Stratify)된 방식으로 분할.
    
    time:
        - 시간을 기준으로 데이터 분할.
        - 특정 기간(holdout_start ~ holdout_end)에 해당하는 데이터를 검증 세트로, 나머지를 학습 세트로 분할.
    """
    if split_type == "random":
        X = train_df.drop(columns=[target_column])  
        y = train_df[target_column]  
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, 
            y, 
            test_size=0.2,
            random_state=42
        )
    elif split_type == "time":
        holdout_start = 202307
        holdout_end = 202312

        x_train = train_df[~((train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end))].drop(columns=[target_column])
        y_train = train_df[~((train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end))][target_column]

        x_valid = train_df[(train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end)].drop(columns=[target_column])
        y_valid = train_df[(train_df["contract_year_month"] >= holdout_start) & (train_df["contract_year_month"] <= holdout_end)][target_column] 
    else:
        raise Exception("Invalid model_name. (data_split.py)")
        
    return x_train, x_valid, y_train, y_valid