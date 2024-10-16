import yaml
import os

def load_params(model_name, model_type):
    possible_model = ["LGBM"]
    possible_type = ["classifier", "regressor"]
    params_path = "models"
    
    if (model_name not in possible_model) or (model_type not in possible_type):
        raise Exception("Invalid model_name or model_type. (load_params.py)")
    else:
        if model_name == "LGBM":
            yaml_load =  yaml.load(open(os.path.join(params_path, "lgbm_param.yaml")), Loader=yaml.FullLoader)
        # elif: 다음 모델
        
        if model_type == "classifier":
            params = yaml_load["cls_params"]
        elif model_type == "regressor":
            params = yaml_load["reg_params"]

        return params