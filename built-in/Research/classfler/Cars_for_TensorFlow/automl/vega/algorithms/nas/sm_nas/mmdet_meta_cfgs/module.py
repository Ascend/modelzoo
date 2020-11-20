"""Base Modules for Meta Config of mmdetection."""

import random
import warnings
from abc import ABCMeta, abstractmethod

from mmcv import Config, ConfigDict

from ..utils.str2dict import str2dict


class Module(metaclass=ABCMeta):
    """Base model of module.

    :param fore_part: fore part
    :type fore_part: dict
    """

    components = [
        'detector',
        'backbone',
        'neck',
        'rpn_head',
        'roi_extractor',
        'shared_head',
        'bbox_head']
    attr_space = dict()  # define the sample space
    quest_dict = dict()  # define all params that need to be questioned.
    module_space = dict()  # define the conflict modules
    id_attrs = []

    def __init__(self, fore_part=None, *args, **kwargs):
        self.model = fore_part
        self.check_module_space()
        self.check_id_attr()

    @property
    def name(self):
        """Return name.

        :return: name
        :rtype: str
        """
        return str(self)

    def __str__(self):
        """Get arch name.

        :return: arch name
        :rtype: str
        """
        return self.__class__.__name__

    @property
    def id_dict(self):
        """Get id dict.

        The dict for identifying module object.
        :return: id dict
        :rtype: dict
        """
        return {key: getattr(self, key) for key in self.id_attrs}

    @property
    @abstractmethod
    def config(self):
        """Get config."""
        pass

    @property
    def dict_config(self):
        """Get config dict.

        :return: condig dict
        :rtype: dict
        """
        cfg_dict = str2dict(self.config).copy()
        cfg_dict.pop('type', None)
        return cfg_dict

    def check_module_space(self):
        """Check module space."""
        for component in self.module_space.keys():
            if component not in self.components:
                raise KeyError(
                    "Module space of {} get a wrong key '{}' (it must be in {}).".format(self.__class__.__name__,
                                                                                         component,
                                                                                         self.components))

    def check_id_attr(self):
        """Check id attr."""
        id_attrs = set(self.id_attrs)
        sample_attrs = set(self.attr_space.keys())
        if not id_attrs >= sample_attrs:
            raise ValueError(
                "{}: all keys of attr_space must be in id_attrs".format(
                    self.__class__.__name__))

    @classmethod
    def quest_from(cls, module, attr):
        """Quest from.

        :param module: module
        :type module: Module
        :param attr: attr
        :type attr: str
        :return: module
        :rtype: dict
        """
        if isinstance(module, Module) is False:
            raise Exception('module is not a Module')
        if hasattr(module, attr):
            return getattr(module, attr)
        else:
            raise AttributeError(
                '{} fail to get {} from {}'.format(cls.__name__, attr, module.__class__.__name__))

    @classmethod
    def quest_param(cls, fore_part=None, **kwargs):
        """Quest params from other modules according to the quest_dict.

        :param fore_part: former modules.
        :return: param dict
        :rtype: dict
        """
        if fore_part is None:
            warnings.warn('There is no fore_part for {}'.format(cls.__name__))
        params = dict()
        for key, value in cls.quest_dict.items():
            given = kwargs.get(key, None)
            if given is None:
                component, attr = value
                params[key] = cls.quest_from(fore_part[component], attr)
            else:
                params[key] = given
        return params

    @classmethod
    def sample(cls, method='random', fore_part=None, **kwargs):
        """Sample a model.

        :param method: search method name
        :type method: str
        :param fore_part: fore part
        :return: sample result
        :rtype: dict
        """
        if method == 'random':
            params = cls.random_sample(fore_part=fore_part, **kwargs)
        elif method == 'EA':
            params = cls.EA_sample(fore_part=fore_part, **kwargs)
        else:
            raise ValueError('Unrecognized sample method {}.'.format(method))
        params.update(cls.quest_param(fore_part=fore_part, **kwargs))
        return cls(**params, fore_part=fore_part)

    @classmethod
    def random_sample(cls, **kwargs):
        """Random sample.

        :return: param dict
        :rtype: dict
        """
        params = dict()
        for key, value in cls.attr_space.items():
            attr = random.choice(value)
            params.update({key: attr})
        return params

    @classmethod
    def EA_sample(cls, fore_part=None, **kwargs):
        """Use ea to sample a model.

        :param fore_part: fore path
        :type fore_part: str
        """
        raise NotImplementedError

    @classmethod
    def set_from_config(cls, config, fore_part=None, **kwargs):
        """Set from config.

        :param config: config
        :type config: dict
        :param fore_part: fore path
        :return: config info
        :rtype: dict)
        """
        if not isinstance(config, (dict, Config, ConfigDict)):
            raise TypeError(
                "{}: 'config' must be a dict or a config, but get a {}.".format(
                    cls.__name__, type(config)))
        config_info = dict(fore_part=fore_part)
        for key in cls.attr_space.keys():
            try:
                config_info.update({key: config.get(key)})
            except BaseException:
                continue
        config_info.update(cls.quest_param(fore_part=fore_part, **kwargs))
        return cls(**config_info)

    @classmethod
    def set_from_model_info(cls, model_info, fore_part=None, **kwargs):
        """Set from model info.

        :param model_info: model info
        :type model_info: dict
        :param fore_part: fore part
        :return: model info
        :rtype: dict
        """
        if not isinstance(model_info, (dict, Config, ConfigDict)):
            raise TypeError(
                "{}: 'config' must be a dict or a config, but get a {}.".format(cls.__name__, type(model_info)))
        model_info.update(fore_part=fore_part)
        model_info.update(cls.quest_param(fore_part=fore_part, **kwargs))
        return cls(**model_info)

    @property
    def sample_result(self):
        """Sample result.

        :return: sample result dict
        :rtype: dict
        """
        sample_result = dict()
        for key in self.attr_space.keys():
            sample_result.update({key: getattr(self, key)})
        return sample_result
