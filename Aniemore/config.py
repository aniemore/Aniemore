import yaml
from os import path


# It reads the config.yml file and makes it available as a dictionary
class Config:
    config_path = path.join(path.dirname(__file__), "config.yml")

    def __init__(self):
        try:
            with open(self.config_path, 'r') as config_file:
                self.configs = yaml.safe_load(config_file)
        except yaml.YAMLError as ex:
            print(ex)

    def __getitem__(self, key):
        return self.configs[key]


config = Config()
