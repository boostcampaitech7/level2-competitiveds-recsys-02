from sklearn.linear_model import LinearRegression, LogisticRegression

class LR:
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
