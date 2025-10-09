# from ai_crack.train.train_config import TrainConfig
# from dataclasses import asdict
# import toml

# config = TrainConfig()

# data_dict = asdict(config)

# toml_str = toml.dumps(data_dict)
# print(toml_str)

# formatted_toml_str = "\n\n".join([line for line in toml_str.splitlines()])

# with open('train_config.toml', 'w', encoding='utf-8') as file:
#     file.write(formatted_toml_str)


from dotmap import DotMap
import toml
import os
from pprint import pprint

thisdir = os.path.dirname(os.path.abspath(__file__))
print(os.path.dirname(thisdir))
toml_path = os.path.join(os.path.dirname(thisdir), "tools/train_config.toml")

with open(toml_path) as f:
    config = DotMap(toml.load(f))

# pprint(config.toDict())
print(config.training.batch_size)
