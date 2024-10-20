from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from catboost import Pool

class Catboost:
    def __init__(self, model_type, params):
        self.model = None
        self.model_type = model_type
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        train_pool = Pool(data=x_train, label=y_train)
        valid_pool = None
        if x_valid is not None and y_valid is not None:
            valid_pool = Pool(data=x_valid, label=y_valid)

        if self.model_type == "classifier":
            model = CatBoostClassifier(**self.params)
        elif self.model_type == "regressor":
            model = CatBoostRegressor(**self.params)

        if valid_pool is None:
            model.fit(train_pool)
        else:
            model.fit(train_pool, eval_set=valid_pool)
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        valid_pool = Pool(data=x_valid)
        y_valid_pred = self.model.predict_proba(valid_pool)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        valid_pool = Pool(data=x_valid)
        pred = self.model.predict(valid_pool)
        return pred
    
    def feature_importance(self):
        if self.model is None:
            raise ValueError("Train first")
        return self.model.feature_importances_