import os
import pandas as pd

def data_loader(dataset_name: str) -> pd.DataFrame:
    # File Load
    data_path: str = "data"
    if dataset_name == "pure":
        drop_columns = ["index", "built_year"]
        target_column = "deposit"
        train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv"))
    elif dataset_name == "data_v1017":
        drop_columns = ["_type"]
        target_column = "deposit"
        dataset_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "merged_data.csv"))
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