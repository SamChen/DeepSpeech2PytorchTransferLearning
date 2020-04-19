import ruamel.yaml
import os
BASIC_DIRS=["tensorboard", "exps", "conf", "data", "decoded"]

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = ruamel.yaml.safe_load(f)
    check_validation(config)
    return config

def check_validation(input):
    assert os.path.isdir(input["basic"]["exp_root_path"]), "{} does not exit".format(input["exp_root_path"])
    exp_root_path = input["basic"]["exp_root_path"]

    for i in BASIC_DIRS:
        file = os.path.join(exp_root_path, i)
        assert os.path.isdir(file), "{} does not exist".format(file)

    for i in input["data"]:
        file = os.path.join(exp_root_path, "data", input["data"][i])
        assert os.path.isfile(file), "{} does not exist".format(file)

    if input["basic"]["augmentation_config_name"]:
        file = os.path.join(exp_root_path, "conf", input["basic"]["augmentation_config_name"])
        assert os.path.isfile(file), "{} does not exist".format(file)

