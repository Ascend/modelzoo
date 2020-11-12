from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = 'V1.0.2'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


ext_modules = [
    Extension(
        'dnmetis_backend',
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/dnmetis_backend/pull/53)
        #sorted(['src/main.cpp','src/model_process.cpp','src/sample_process.cpp','src/utils.cpp']),
        sources=[
            'src/main.cpp',
            'src/Config.cpp',
            'backend/BaseBackend.cpp',
            'backend/BackendFactory.cpp',

            'backend/built-in/aclbackend.cpp',
            'backend/built-in/src/model_process.cpp',
            'backend/built-in/src/sample_process.cpp',
            'backend/built-in/src/utils.cpp',

            'backend/custom/trtbackend.cpp',
        ],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            './inc/',
            './backend/inc',
            './backend/built-in',
            './backend/built-in/inc',
            './backend/custom',
            './backend/custom/inc',
            '/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023//acllib/include/',
        ],
        library_dirs=['/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023//acllib/lib64/',],
        libraries=['ascendcl',],
        language='c++'
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++14', '-std=c++17','-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-w','-O0', '-fpermissive'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-O0'],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name='dnmetis_backend',
    version=__version__,
    author='chegulu',
    author_email='chegulu@xxx.com',
    url='https://chegulu',
    description='A test tool project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11==2.5.0'],
    #data_files=[('config', ['cfg/config.txt'])],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
