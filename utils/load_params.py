import yaml
import os

def load_params(model_name, model_type):
    """
    특정 모델과 유형에 맞는 하이퍼파라미터를 YAML 파일에서 불러옴.

    Args:
        model_name (str): 불러올 모델 이름, "LGBM", "XGB", "Catboost", "RF", "LR" 중 하나여야 함.
        model_type (str): 모델 유형, "classifier" 또는 "regressor" 중 하나여야 함.

    Raises:
        Exception: model_name 또는 model_type이 허용되지 않은 값일 때 예외 발생.

    Returns:
        dict: 모델에 맞는 하이퍼파라미터 딕셔너리. 'classifier'의 경우 분류 모델 파라미터를, 
              'regressor'의 경우 회귀 모델 파라미터를 반환.
    """
    possible_model = ["LGBM", "XGB", "Catboost", "RF", "LR"]
    possible_type = ["classifier", "regressor"]
    params_path = "models/params"
    
    if (model_name not in possible_model) or (model_type not in possible_type):
        raise Exception("Invalid model_name or model_type. (load_params.py)")
    else:
        if model_name == "LGBM":
            yaml_load =  yaml.load(open(os.path.join(params_path, "lgbm_param.yaml")), Loader=yaml.FullLoader)
        elif model_name == "XGB":
            yaml_load =  yaml.load(open(os.path.join(params_path, "xgb_param.yaml")), Loader=yaml.FullLoader)
        elif model_name == "Catboost":
            yaml_load =  yaml.load(open(os.path.join(params_path, "catboost_param.yaml")), Loader=yaml.FullLoader)
        elif model_name == "RF":
            yaml_load =  yaml.load(open(os.path.join(params_path, "rf_param.yaml")), Loader=yaml.FullLoader)
        elif model_name == "LR":
            yaml_load =  yaml.load(open(os.path.join(params_path, "lr_param.yaml")), Loader=yaml.FullLoader)
        
        if model_type == "classifier":
            params = yaml_load["cls_params"]
        elif model_type == "regressor":
            params = yaml_load["reg_params"]

        return params