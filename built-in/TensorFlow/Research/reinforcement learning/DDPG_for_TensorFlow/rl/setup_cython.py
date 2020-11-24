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
