# -*- coding:utf-8 -*-

''''''

import typing

from collections import defaultdict


class Factory(object):
    ''''''

    def __init__(self) -> None:
        ''''''

        self.construct_map = defaultdict(dict)

    def register(self, type: str, name: str, construct_func: typing.Callable[..., typing.Any]) -> None:
        ''''''

        self.construct_map[type][name] = construct_func

    def create(self, type: str, name: str, *args, **kwargs):
        ''''''

        if name not in self.construct_map:
            raise NotImplementedError(f'Factory does not support {name}')

        return self.construct_map[type][name](*args, **kwargs)

FACTORY = Factory()

def REGISTER(type: str, name: str, construct_func: typing.Callable[..., typing.Any]) -> None:
    ''''''

    FACTORY.register(type, name, construct_func)

def CREATE(type: str, name: str, *args, **kwargs):
    ''''''

    return FACTORY.create(type, name, *args, **kwargs)
