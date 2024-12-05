from models.lgbm import LGBM
from models.xgb import XGB
from models.catboost import Catboost
from models.randomforest import RF
from models.linear_regression import LR

def train_model(model_name, model_type, params, x_train, y_train):
    """
    주어진 모델과 유형에 맞는 모델을 학습시킴.

    Args:
        model_name (str): 학습할 모델 이름, "LGBM", "XGB", "Catboost", "RF", "LR" 중 하나여야 함.
        model_type (str): 모델 유형, "classifier" 또는 "regressor".
        params (dict): 모델 학습에 사용할 하이퍼파라미터.
        x_train (pd.DataFrame): 학습에 사용할 입력 데이터.
        y_train (pd.Series): 학습에 사용할 타겟 데이터.

    Raises:
        Exception: model_name 또는 model_type이 허용되지 않은 값일 경우 예외 발생.

    Returns:
        model: 학습된 모델 객체.

    설명:
        - 모델 이름과 유형에 맞는 모델 객체를 생성.
        - 주어진 학습 데이터(x_train, y_train)를 사용하여 모델을 학습.
    """
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