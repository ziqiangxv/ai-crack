# -*- coding:utf-8 -*-

''''''

import os
from setuptools import find_packages
from setuptools import setup

this_dir = os.path.abspath(os.path.dirname(__file__))

version = '0.0.0+debug'
with open(os.path.join(this_dir, '.version'),encoding='utf-8') as f:
    version = f.readlines()[0]

description = ''
with open(os.path.join(this_dir, 'README.md'),encoding='utf-8') as f:
    description = ''.join(f.readlines())

url = 'https://github.com/ziqiangxv/ai-crack.git'

with open(os.path.join(this_dir, 'requirements.txt'),encoding='utf-8') as f:
    requirements = [l.replace('\n', '') for l in f.readlines()]

setup(
    name = "ai_crack",
    version = version,
    description = description,
    url = url,
    download_url = url,
    author = "1079602703@qq.com",
    install_requires = requirements,
    packages = find_packages(),
    # entry_points={
    #     'console_scripts': [
    #         'ppl_test = console_scripts.image_pipeline_test:ppl_test',  # snek 是命令名，snek:main 指定了模块和函数
    #     ],
    # },
)
