import copy
import importlib
from .geometric import *
from .photometric import *



def build_augmenter(config, additional_config=None):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    augmenter = []

    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]
        
        if not value:
            value = {}
            
        if additional_config:
            for add_name, add_value in additional_config.items():
                if name == str(add_name) and eval(name):
                    value.update(add_value)

        arch = getattr(mod, name)(**value)
        augmenter.append(arch)
    return augmenter