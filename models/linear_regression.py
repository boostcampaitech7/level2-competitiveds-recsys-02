from sklearn.linear_model import LinearRegression, LogisticRegression

class LR:
    """
    선형 회귀 및 로지스틱 회귀 모델을 다루는 클래스. 모델을 학습하고 예측하며, 특성 중요도를 반환.

    Args:
        model_type (str): 모델 유형, "classifier" 또는 "regressor".
        params (dict): 회귀 모델 학습에 필요한 하이퍼파라미터.

    Methods:
        train: 학습 데이터를 사용하여 모델을 학습.
            Args:
                x_train (pd.DataFrame): 학습 입력 데이터.
                y_train (pd.Series): 학습 타겟 데이터.
                x_valid (pd.DataFrame, optional): 검증 입력 데이터.
                y_valid (pd.Series, optional): 검증 타겟 데이터.
            - 검증 데이터가 주어지면 이를 사용하여 모델을 학습하며, 주어지지 않으면 학습 데이터만으로 학습.
        
        predict_proba: 검증 데이터에 대한 클래스 확률 예측 (분류 모델에서만 사용 가능).
            Args:
                x_valid (pd.DataFrame): 검증 데이터.
            Returns:
                np.ndarray: 각 클래스에 대한 확률 예측 값.
            Raises:
                ValueError: 분류 모델이 아닐 경우 예외 발생.
        
        predict: 검증 데이터에 대한 예측.
            Args:
                x_valid (pd.DataFrame): 검증 데이터.
            Returns:
                np.ndarray: 예측된 값.

        feature_importance: 학습된 모델의 특성 중요도를 반환.
            Returns:
                np.ndarray: 각 특성에 대한 중요도 값.
            - 회귀 모델의 경우 계수가 중요도, 분류 모델에서도 계수가 중요도로 사용됨.
    """
    def __init__(self, model_type, params):
        self.model = None
        self.model_type = model_type
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        if self.model_type == "classifier":
            model = LogisticRegression(**self.params)
        elif self.model_type == "regressor":
            model = LinearRegression(**self.params)
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        if self.model_type == "classifier":
            y_valid_pred = self.model.predict_proba(x_valid)
            return y_valid_pred
        else:
            raise ValueError("predict_proba is only available for classification models.")
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        pred = self.model.predict(x_valid)
        return pred
    
    def feature_importance(self):
        if self.model is None:
            raise ValueError("Train first")
        if self.model_type == "regressor":
            return self.model.coef_  # For Linear Regression, feature importances are the coefficients
        elif self.model_type == "classifier":
            return self.model.coef_  # For Logistic Regression, coefficients represent feature importances
