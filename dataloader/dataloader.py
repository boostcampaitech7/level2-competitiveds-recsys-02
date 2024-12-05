import os
import pandas as pd

def data_loader(dataset_name: str) -> pd.DataFrame:
    """
    주어진 데이터셋 이름에 따라 학습 및 테스트 데이터를 로드하는 함수.

    Args:
        dataset_name (str): 로드할 데이터셋의 이름. "pure", "data_v1017", "data_v1021", "final_df" 중 하나.

    Returns:
        tuple: 
            - train_df (pd.DataFrame): 학습 데이터프레임.
            - test_df (pd.DataFrame): 테스트 데이터프레임.
            - submission_df (pd.DataFrame): 제출 파일을 위한 데이터프레임.
            - drop_columns (list): 제거된 열의 이름 목록.
            - target_column (str): 타겟 변수의 이름.

    Raises:
        FileNotFoundError: 지정된 데이터 파일이 존재하지 않는 경우 발생.

    - 데이터 로드 후 전처리 작업으로, 특정 열을 제거하고, 결측치를 처리하며, 열 이름에서 특수 문자를 제거.
    """
    data_path: str = "data"
    if dataset_name == "pure":
        drop_columns = ["index", "built_year"]
        target_column = "deposit"
        train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv"))
    elif dataset_name == "data_v1017":
        drop_columns = ["_type"]
        target_column = "deposit"
        dataset_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "merged_data_v1017.csv"))
        train_df = dataset_df.loc[dataset_df["_type"]=="train"]
        test_df = dataset_df.loc[dataset_df["_type"]=="test"]
        test_df = test_df.drop(columns=[target_column])
    elif dataset_name == "data_v1021":
        drop_columns = ["_type", "deposit"]
        target_column = "deposit_by_area"
        dataset_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "merged_data_v1021.csv"))
        train_df = dataset_df.loc[dataset_df["_type"]=="train"]
        test_df = dataset_df.loc[dataset_df["_type"]=="test"]
        test_df = test_df.drop(columns=[target_column])
    elif dataset_name == "final_df":
        drop_columns = ["_type", "deposit"]
        target_column = "deposit_by_area"
        dataset_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "final_df.csv"))
        train_df = dataset_df.loc[dataset_df["_type"]=="train"]
        test_df = dataset_df.loc[dataset_df["_type"]=="test"]
        test_df = test_df.drop(columns=[target_column])
    elif dataset_name == "real_final_df":
        drop_columns = ["_type", "deposit_by_area", "subways_within_1km", "park_count_500m", "subways_within_500m", "Is_Outside", "park_distance_kurtosis", "park_distance_skewness"]
        target_column = "deposit"
        dataset_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "final_df.csv"))
        train_df = dataset_df.loc[dataset_df["_type"]=="train"]
        test_df = dataset_df.loc[dataset_df["_type"]=="test"]
        test_df = test_df.drop(columns=[target_column])
    
    
    submission_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    
    train_df = train_df.drop(columns=drop_columns)
    test_df = test_df.drop(columns=drop_columns)

    train_df.columns = train_df.columns.str.replace(r'[^\w\s]', '', regex=True)
    test_df.columns = test_df.columns.str.replace(r'[^\w\s]', '', regex=True)

    train_df = train_df.ffill().fillna(-999)
    test_df = test_df.ffill().fillna(-999)
    
    return train_df, test_df, submission_df , drop_columns, target_column