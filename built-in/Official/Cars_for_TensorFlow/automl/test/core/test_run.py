# -*- coding:utf-8 -*-
"""Test function for vega.run"""
import unittest


def lazy(func):
    """lazy function wrapper

    :param func: function name
    """
    attr_name = "_lazy_" + func.__name__

    def lazy_func(*args, **kwargs):
        """Wrapper of lazy func

        :param args: any object
        :param kwargs: any kwargs
        :return:
        """
        if not hasattr(func, attr_name):
            setattr(func, attr_name, func(*args, **kwargs))
        return getattr(func, attr_name)

    return lazy_func


@lazy
def env_args(args):
    """A lazy function will be execute when call

    :param args: any object
    :return:
    """
    return args


class TestPipeline(unittest.TestCase):
    """Test lazy function worked in pipeline"""

    def test_env_args(self):
        """Test function 'env_args' is a lazy function"""
        args = {'env': 'test'}
        env_args(args)
        self.assertEqual(env_args(), {'env': 'test'})
