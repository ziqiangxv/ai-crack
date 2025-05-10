# -*- coding:utf-8 -*-

''''''

def color_print(s: str, color: str = 'red'):
    if color in ('red', 'r'):
        print(f'\033[31m{s}\033[0m')

    elif color in ('green', 'g'):
        print(f'\033[92m{s}\033[0m')

    elif color in ('blue', 'b'):
        print(f'\033[34m{s}\033[0m')

    else:
        print(s)

def color(s: str, color: str = 'red'):
    if color in ('red', 'r'):
        return f'\033[31m{s}\033[0m'

    elif color in ('green', 'g'):
        return f'\033[92m{s}\033[0m'

    elif color in ('blue', 'b'):
        return f'\033[34m{s}\033[0m'

    else:
        return s

