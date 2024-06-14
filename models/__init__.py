import copy
import importlib
from .scene_text_detection import STD
from .craft import CRAFT
from .architectures import VGG16_backbone, VGG19_backbone
from .layers import GaussianBuilder


def build_models(config, weights=None):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    input_shape = config.pop("input_shape")
    weight_path = config.pop("weight_path")
    load_weight_type = config.pop("load_weight_type")

    architecture_config = config['Architecture']
    architecture_name = architecture_config.pop("name")
    
    sub_model = None
    if architecture_name.lower() == "craft":
        sub_model_config = config['Perspective_Transfrom']
        sub_model_name = sub_model_config.pop("name")
        sub_model = getattr(mod, sub_model_name)(**sub_model_config)
        
    backbone_config = config['Backbone']
    backbone_config['input_shape'] = [None, None, input_shape[-1]]
    backbone_name = backbone_config.pop("name")
    backbone = getattr(mod, backbone_name)(**backbone_config)
    architecture_config['backbone'] = backbone
    architecture = getattr(mod, architecture_name)(**architecture_config)
    model = STD(architecture)

    if weights:
        model.load_weights(weights)
    else:
        if load_weight_type and weight_path:
            if load_weight_type == "weights":
                model.load_weights(weight_path)
            elif load_weight_type == "models":
                model.load_models(weight_path)
    return model, sub_model
