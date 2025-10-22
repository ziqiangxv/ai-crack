# -*- coding:utf-8 -*-

''''''

import os
import torch
torch.manual_seed(3407)

from ai_crack import Trainer, TrainConfig

this_dir = os.path.dirname(os.path.abspath(__file__))
train_config_path = os.path.join(this_dir, 'train_config.toml')
config = TrainConfig(train_config_path)

trainer = Trainer(config)
trainer.train(epoch_prints = 10)

