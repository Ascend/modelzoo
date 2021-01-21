"""
Misc
"""


def pair(x, dims=2):
    if isinstance(x, list) or isinstance(x, tuple):
        assert len(x) == dims
    elif isinstance(x, int):
        x = [x] * dims
    else:
        raise ValueError
    return x
