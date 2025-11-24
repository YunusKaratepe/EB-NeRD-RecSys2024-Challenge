import yaml

def read_yaml(path):
    with open(path, 'r') as file:
        content = yaml.safe_load(file)
    return content