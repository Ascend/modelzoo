# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
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

"""Registry for encoders, decoder, attentions and models.

Define a new model and register it:

```
@registry.register_model
class MyModel():
  ...
```

Access by snake-cased name: `registry.model("my_model")`
See all the classes registered: `registry.list_models()`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import re
import six
import tensorflow as tf

_MODELS = {}
_ENCODERS = {}
_DECODERS = {}
_ATTENTIONS = {}
_CLASSES = {}

# Camel case to snake case utils
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def _convert_camel_to_snake(name):
  s1 = _first_cap_re.sub(r"\1_\2", name)
  return _all_cap_re.sub(r"\1_\2", s1).lower()


def _reset():
  for ctr in [_MODELS, _ENCODERS, _DECODERS, _ATTENTIONS]:
    ctr.clear()


def default_name(obj_class):
  """Convert a class name to the registry's default name for the class.

  Args:
    obj_class: the name of a class

  Returns:
    The registry's default name for the class.
  """
  return _convert_camel_to_snake(obj_class.__name__)


def default_object_name(obj):
  """Convert an object to the registry's default name for the object class.

  Args:
    obj: an object instance

  Returns:
    The registry's default name for the class of the object.
  """
  return default_name(obj.__class__)


def _register_cls(name, DICT):
  """Register a class. name defaults to class name snake-cased."""

  def decorator(cls_, registration_name=None):
    """Registers & returns model_cls with registration_name or default name."""
    cls_name = registration_name or default_name(cls_)
    if cls_name in DICT and not tf.contrib.eager.in_eager_mode():
      raise LookupError("Class %s already registered." % cls_name)
    cls_.REGISTERED_NAME = cls_name
    DICT[cls_name] = cls_
    return cls_

  # Handle if decorator was used without parens
  if callable(name):
    cls_ = name
    return decorator(cls_, registration_name=default_name(cls_))

  return lambda cls_: decorator(cls_, name)


### register general classes
def register_class(name=None):
  """Register a model. name defaults to class name snake-cased."""
  return _register_cls(name, _CLASSES)


def class_ins(name):
  name = _convert_camel_to_snake(name)
  if name not in _CLASSES:
    raise LookupError("Class %s never registered.  Available classes:\n %s" %
                      (name, "\n".join(list_classes())))

  return _CLASSES[name]


def list_classes():
  return list(sorted(_CLASSES))


### register model
def register_model(name=None):
  """Register a model. name defaults to class name snake-cased."""
  return _register_cls(name, _MODELS)


def model(name):
  name = _convert_camel_to_snake(name)
  if name not in _MODELS:
    raise LookupError("Model %s never registered.  Available models:\n %s" %
                      (name, "\n".join(list_models())))

  return _MODELS[name]


def list_models():
  return list(sorted(_MODELS))


### register encoder
def register_encoder(name=None):
  """Register a model. name defaults to class name snake-cased."""
  return _register_cls(name, _ENCODERS)


def encoder(name):
  name = _convert_camel_to_snake(name)
  if name not in _ENCODERS:
    raise LookupError("Model %s never registered.  Available models:\n %s" %
                      (name, "\n".join(list_encoders())))

  return _ENCODERS[name]


def list_encoders():
  return list(sorted(_ENCODERS))


### register decoder
def register_decoder(name=None):
  """Register a model. name defaults to class name snake-cased."""
  return _register_cls(name, _DECODERS)


def decoder(name):
  name = _convert_camel_to_snake(name)
  if name not in _DECODERS:
    raise LookupError("Model %s never registered.  Available models:\n %s" %
                      (name, "\n".join(list_decoders())))

  return _DECODERS[name]


def list_decoders():
  return list(sorted(_DECODERS))


### register attention
def register_attention(name=None):
  """Register a model. name defaults to class name snake-cased."""
  return _register_cls(name, _ATTENTIONS)


def attention(name):
  name = _convert_camel_to_snake(name)
  if name not in _ATTENTIONS:
    raise LookupError("Model %s never registered.  Available models:\n %s" %
                      (name, "\n".join(list_attentions())))

  return _ATTENTIONS[name]


def list_attentions():
  return list(sorted(_ATTENTIONS))


def display_list_by_prefix(names_list, starting_spaces=0):
  """Creates a help string for names_list grouped by prefix."""
  cur_prefix, result_lines = None, []
  space = " " * starting_spaces
  for name in sorted(names_list):
    split = name.split("_", 1)
    prefix = split[0]
    if cur_prefix != prefix:
      result_lines.append(space + prefix + ":")
      cur_prefix = prefix
    result_lines.append(space + "  * " + name)
  return "\n".join(result_lines)


def help_string():
  """Generate help string with contents of registry."""
  help_str = """
Registry contents:
------------------

  Models:
%s

  Encoders:
%s

  Decoders:
%s

  Attentions:
%s
  Other classes:
%s
"""
  m, enc, dec, att, cls_ = [
      display_list_by_prefix(entries, starting_spaces=4) for entries in [
          list_models(),
          list_encoders(),
          list_decoders(),
          list_attentions(),
          list_classes(),
      ]
  ]
  return help_str % (m, enc, dec, att, cls_)
