from models.lgbm import LGBM
from models.xgb import XGB
from models.catboost import Catboost
from models.randomforest import RF
from models.linear_regression import LR

def train_model(model_name, model_type, params, x_train, y_train):
    possible_model = ["LGBM", "XGB", "Catboost", "RF", "LR"]
    possible_type = ["classifier", "regressor"]

    if (model_name not in possible_model) or (model_type not in possible_type):
        raise Exception("Invalid model_name or model_type. (train_mpdel.py)")
    else:
        if model_name == "LGBM":
            model = LGBM(model_type, params)
        elif model_name == "XGB":
            model = XGB(model_type, params)
        elif model_name == "Catboost":
            model = Catboost(model_type, params)
        elif model_name == "RF":
            model = RF(model_type, params)
        elif model_name == "LR":
            model = LR(model_type, params)
        # elif: 다음 모델
                    
    
    model.train(x_train, y_train)

    return model