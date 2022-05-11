from .exception import XAIConfigException

gan_labels = {
    'real': 1.,
    'fake': 0.
}

def check_for_key(config, key, value=None):
    if value is None:
        if config is not None and key in config.keys():
            return config[key]
        else:
            return None
    else:
        if config is not None and key in config.keys():
            if config[key] == value:
                return True
            else:
                raise Exception(f'Unknown parameter \'{key}\' in explanation_config : {value}')
        else:
            return False
