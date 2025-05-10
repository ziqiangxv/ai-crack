from ai_crack.train.train_config import TrainConfig
from dataclasses import asdict
import toml

config = TrainConfig()

data_dict = asdict(config)

toml_str = toml.dumps(data_dict)
print(toml_str)

formatted_toml_str = "\n\n".join([line for line in toml_str.splitlines()])

with open('train_config.toml', 'w', encoding='utf-8') as file:
    file.write(formatted_toml_str)
