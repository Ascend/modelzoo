# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from distutils.core import setup
from Cython.Build import cythonize

cython_pkgs = []
module_entry = "xt"
module_status = False

for root, dirs, files in os.walk(module_entry):
    for file in files:
        if file == "__init__.py":
            module_status = True
    if module_status is True:
        cython_pkgs.append(root + "/*.py")
    module_status = False

print(cython_pkgs)

setup(
    name="xingtian",
    ext_modules=cythonize(
        cython_pkgs, compiler_directives={"language_level": 3, "embedsignature": True}
    ),
)

for root, dirs, files in os.walk(module_entry):
    # os.system("rm " + root + "/*.so")
    os.system("rm " + root + "/*.c")
