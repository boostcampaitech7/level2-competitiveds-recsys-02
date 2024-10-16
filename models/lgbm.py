from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

class LGBM:
    def __init__(self, model_type, params):
        self.model = None
        self.model_type = model_type
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        if self.model_type == "classifier":
            model = LGBMClassifier(**self.params)
        elif self.model_type == "regressor":
            model = LGBMRegressor(**self.params)
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        y_valid_pred = self.model.predict_proba(x_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        pred = self.model.predict(x_valid)
        return pred