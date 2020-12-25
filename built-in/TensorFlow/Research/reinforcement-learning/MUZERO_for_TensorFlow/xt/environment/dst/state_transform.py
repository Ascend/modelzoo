# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict

import numpy as np
import gym

VALIDATION_INTERVAL = 100


class Preprocessor:
    """Defines an abstract observation preprocessor function.
    Attributes:
        shape (obj): Shape of the preprocessed output.
    """

    def __init__(self, obs_space):
        legacy_patch_shapes(obs_space)
        self._obs_space = obs_space
        self.shape = self._init_shape(obs_space)
        self._size = int(np.product(self.shape))
        self._i = 0

    def _init_shape(self, obs_space):
        """Returns the shape after preprocessing."""
        raise NotImplementedError

    def transform(self, observation):
        """Returns the preprocessed observation."""
        raise NotImplementedError

    def write(self, observation, array, offset):
        """Alternative to transform for more efficient flattening."""
        array[offset:offset + self._size] = self.transform(observation)

    def check_shape(self, observation):
        """Checks the shape of the given observation."""
        if self._i % VALIDATION_INTERVAL == 0:
            if type(observation) is list and isinstance(
                    self._obs_space, gym.spaces.Box):
                observation = np.array(observation)
            try:
                if not self._obs_space.contains(observation):
                    raise ValueError(
                        "Observation outside expected value range",
                        self._obs_space, observation)
            except AttributeError:
                raise ValueError(
                    "Observation for a Box/MultiBinary/MultiDiscrete space "
                    "should be an np.array, not a Python list.", observation)
        self._i += 1

    @property
    def size(self):
        return self._size


class OneHotPreprocessor(Preprocessor):
    def _init_shape(self, obs_space):
        return (self._obs_space.n, )

    def transform(self, observation):
        self.check_shape(observation)
        arr = np.zeros(self._obs_space.n, dtype=np.float32)
        arr[observation] = 1
        return arr

    def write(self, observation, array, offset):
        array[offset + observation] = 1


class NoPreprocessor(Preprocessor):
    def _init_shape(self, obs_space):
        return self._obs_space.shape

    def transform(self, observation):
        self.check_shape(observation)
        return observation

    def write(self, observation, array, offset):
        array[offset:offset + self._size] = np.array(
            observation, copy=False).ravel()

    def observation_space(self):
        return self._obs_space


class TupleFlatteningPreprocessor(Preprocessor):
    """
    Preprocesses each tuple element, then flattens it all into a vector.
    """

    def _init_shape(self, obs_space):
        assert isinstance(self._obs_space, gym.spaces.Tuple)
        size = 0
        self.preprocessors = []
        for i in range(len(self._obs_space.spaces)):
            space = self._obs_space.spaces[i]
            preprocessor = get_preprocessor(space)(space)
            self.preprocessors.append(preprocessor)
            size += preprocessor.size
        return (size, )

    def transform(self, observation):
        self.check_shape(observation)
        array = np.zeros(self.shape)
        self.write(observation, array, 0)
        return array

    def write(self, observation, array, offset):
        assert len(observation) == len(self.preprocessors), observation
        for o, p in zip(observation, self.preprocessors):
            p.write(o, array, offset)
            offset += p.size


class DictFlatteningPreprocessor(Preprocessor):
    """
    Preprocesses each dict value, then flattens it all into a vector.
    """
    def _init_shape(self, obs_space):
        assert isinstance(self._obs_space, gym.spaces.Dict)
        size = 0
        self.preprocessors = []
        for space in self._obs_space.spaces.values():
            preprocessor = get_preprocessor(space)(space)
            self.preprocessors.append(preprocessor)
            size += preprocessor.size
        return (size, )

    def transform(self, observation):
        self.check_shape(observation)
        array = np.zeros(self.shape)
        self.write(observation, array, 0)
        return array

    def write(self, observation, array, offset):
        if not isinstance(observation, OrderedDict):
            observation = OrderedDict(sorted(observation.items()))
        assert len(observation) == len(self.preprocessors), \
            (len(observation), len(self.preprocessors))
        #print(self.preprocessors)
        for o, p in zip(observation.values(), self.preprocessors):
            p.write(o, array, offset)
            offset += p.size


def get_preprocessor(space):
    """Returns an appropriate preprocessor class for the given space."""
    if isinstance(space, gym.spaces.Discrete):
        preprocessor = OneHotPreprocessor
    elif isinstance(space, gym.spaces.Tuple):
        preprocessor = TupleFlatteningPreprocessor
    elif isinstance(space, gym.spaces.Dict):
        preprocessor = DictFlatteningPreprocessor
    else:
        preprocessor = NoPreprocessor

    return preprocessor


def legacy_patch_shapes(space):
    """Assigns shapes to spaces that don't have shapes.

    This is only needed for older gym versions that don't set shapes properly
    for Tuple and Discrete spaces.
    """

    if not hasattr(space, "shape"):
        if isinstance(space, gym.spaces.Discrete):
            space.shape = ()
        elif isinstance(space, gym.spaces.Tuple):
            shapes = []
            for s in space.spaces:
                shape = legacy_patch_shapes(s)
                shapes.append(shape)
            space.shape = tuple(shapes)

    return space.shape
