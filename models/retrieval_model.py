from sklearn.preprocessing import StandardScaler
import faiss
import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm

class RetrievalModel:
    """
    A class used to represent a Retrieval Model for deposit prediction.
    
    Attributes
    -------
    feature_weight : Dict
        A dictionary containing feature weights for scaling.
    max_faiss_num : int
        Maximum number of nearest neighbors to retrieve using FAISS.
    l2_limit : int
        L2 distance limit for nearest neighbors.
    rate_weight : float
        Weight for the rate of change in deposit.
    sub_model : Any
        A sub-model for prediction if no matching data is found.
    deposit_mean_df : pd.DataFrame
        DataFrame containing mean deposit values for different locations and dates.
    scaler : StandardScaler
        Scaler for normalizing the data.
    train_df : pd.DataFrame
        DataFrame containing the training data.
    test_df : pd.DataFrame
        DataFrame containing the test data.
        
    Methods
    -------
    fit_scaler(train_df)
        Fits the scaler to the training data.
    get_rate(start_date, X)
        Calculates the rate of change in the deposit between two dates.
    retrieval(X)
        Retrieves the predicted deposit value for the given input data.
    """
    def __init__(self , feature_weight: Dict = {"area_m2": 10},  max_faiss_num: int = 10, l2_limit: float = 3, rate_weight: float = 1, model = None, deposit_mean_df: pd.DataFrame = None):
        """
        Initializes the retrieval model with the given parameters.

        Args:
            feature_weight (Dict, optional): A dictionary specifying the weights for different features. Defaults to {"area_m2": 10}.
            max_faiss_num (int, optional): The maximum number of FAISS neighbors to retrieve. Defaults to 10.
            l2_limit (float, optional): The L2 distance limit for retrieval. Defaults to 3.
            rate_weight (float, optional): The weight for the rate feature. Defaults to 1.
            model (optional): The sub-model to be used. Defaults to None.
            deposit_mean_df (pd.DataFrame, optional): A DataFrame containing the mean deposit values. Defaults to None.

        Attributes:
            train_df (pd.DataFrame): DataFrame for training data. Initialized as None.
            test_df (pd.DataFrame): DataFrame for testing data. Initialized as None.
            feature_weight (Dict): Weights for different features.
            max_faiss_num (int): Maximum number of FAISS neighbors to retrieve.
            l2_limit (float): L2 distance limit for retrieval.
            rate_weight (float): Weight for the rate feature.
            sub_model: The sub-model to be used.
            deposit_mean_df (pd.DataFrame): DataFrame containing the mean deposit values.
            scaler (StandardScaler): Scaler for standardizing features.
        """
        self.train_df = None
        self.test_df = None
        self.feature_weight = feature_weight
        self.max_faiss_num = max_faiss_num
        self.l2_limit = l2_limit
        self.rate_weight = rate_weight
        self.sub_model = model
        self.deposit_mean_df = deposit_mean_df
        self.scaler = StandardScaler()
    
    # fit_scaler function fits the scaler to the train data.
    def fit_scaler(self, train_df: pd.DataFrame) -> None:
        """
        Fits the scaler to the training data excluding the 'deposit' column.

        Parameters:
        train_df (pd.DataFrame): The training data containing the 'deposit' column.

        Returns:
        None
        """
        self.scaler.fit(train_df.drop(columns=['deposit']))
        self.train_df = train_df
    
    # get_rate function calculates the rate of change in the deposit between two dates.
    def get_rate(self, start_date, X):
        if self.deposit_mean_df is None:
            return 0
        
        end_date = X['contract_year_month'].values[0]
        lat_lon = f"{X['latitude'].values[0]}_{X['longitude'].values[0]}"
        start_date = str(int(start_date))
        end_date = str(int(end_date))
        end_deposit_mean = self.deposit_mean_df.loc[lat_lon, end_date]
        start_deposit_mean = self.deposit_mean_df.loc[lat_lon, start_date]
        
        # If the deposit mean is not available, return 0
        if pd.isna(end_deposit_mean) or pd.isna(start_deposit_mean):
            return 0
        
        return (end_deposit_mean - start_deposit_mean) / start_deposit_mean
    
    # get_rate function calculates the rate of change in the deposit between two dates.
    def retrieval(self, X: pd.DataFrame) -> None:
        if self.train_df is None:
            raise ValueError('run fit_scaler first')
        lat, lon = X['latitude'].values[0], X['longitude'].values[0]
        filtered_train_data = self.train_df[(self.train_df['latitude'] == lat) & (self.train_df['longitude'] == lon)]
        if not filtered_train_data.empty:
            filtered_train_y = filtered_train_data['deposit']
            filtered_train_data_scaled = self.scaler.transform(filtered_train_data.drop(columns=['deposit']))
            
            # Scale the input data
            X_scaled = self.scaler.transform(X)
            
            # Multiply each feature by the weight
            for column_name, weight in self.feature_weight.items():
                filtered_train_data_scaled[:, filtered_train_data.columns.get_loc(column_name)] *= weight
                X_scaled[:, X.columns.get_loc(column_name)] *= weight
            
            # Create an index and search for the nearest neighbors
            index = faiss.IndexFlatL2(filtered_train_data_scaled.shape[1])
            index.add(filtered_train_data_scaled)
            D, I = index.search(X_scaled, self.max_faiss_num)
            result_pred = []
            
            for i in range(self.max_faiss_num):
                if i != 0 and D[0][i] > self.l2_limit:
                    break
                
                # Calculate the rate of change in the deposit
                rate = self.get_rate(filtered_train_data.iloc[I[0][i]]['contract_year_month'], X)
                
                result = filtered_train_y.iloc[I[0][i]] * (1 + rate * self.rate_weight)
                result_pred.append(result)
                
            pred = np.mean(result_pred)
            return pred
        
        # sub_model is used if no matching data is found
        else:
            if self.sub_model == None:
                return 0
            else:
                return self.sub_model.predict([X])

    def predict(self, train_df:pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        self.fit_scaler(train_df)
        submission = []
        for i in tqdm(range(len(X))):
            submission.append(self.retrieval(X.iloc[[i]]))
            
        return np.array(submission)
    

def retrieve_model(train_df, deposit_mean_df):
    model = RetrievalModel(deposit_mean_df=deposit_mean_df)
    model.fit_scaler(train_df)
    return model


if __name__ == "__main__":
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    submission_df = pd.read_csv('./data/sample_submission.csv')
    deposit_mean_df = pd.read_csv('./data/deposit_mean_df.csv', index_col=0)
    model = retrieve_model(train_df, deposit_mean_df)
    preds = model.predict(train_df, test_df)    
    submission_df['deposit'] = preds
    submission_df.to_csv('./data/retrieval_submission.csv', index=False)
