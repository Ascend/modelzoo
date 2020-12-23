## dnmetis_backend

  It contains one AclBackend(C++), It can be called by DNMetis when tester tests NPU perf&accuracy
  
  The third-party C++ Backends can be easily added 

1.install dnmetis_backend

    python3.7.5 setup.py  install
Notice that in the setup.py (line 49~51), If you install toolkit in a different path，please modify the paths of libs and includes :
```
            '/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023//acllib/include/',
        ],
        library_dirs=['/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023//acllib/lib64/',],
```

2.Install log:

/usr/local/python3.7.5/lib/python3.7/site-packages/setuptools/dist.py:474: UserWarning: Normalizing 'V1.0.2' to '1.0.2'
  normalized_version,
running install
running bdist_egg
running egg_info
creating dnmetis_backend.egg-info
writing dnmetis_backend.egg-info/PKG-INFO
writing dependency_links to dnmetis_backend.egg-info/dependency_links.txt
writing top-level names to dnmetis_backend.egg-info/top_level.txt
writing manifest file 'dnmetis_backend.egg-info/SOURCES.txt'
reading manifest file 'dnmetis_backend.egg-info/SOURCES.txt'
writing manifest file 'dnmetis_backend.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_ext
creating tmp
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/usr/local/python3.7.5/include/python3.7m -c /tmp/tmp4o2xo183.cpp -o tmp/tmp4o2xo183.o -std=c++14
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/usr/local/python3.7.5/include/python3.7m -c /tmp/tmptazg09at.cpp -o tmp/tmptazg09at.o -fvisibility=hidden
building 'dnmetis_backend' extension
creating build
creating build/temp.linux-x86_64-3.7
creating build/temp.linux-x86_64-3.7/src
creating build/temp.linux-x86_64-3.7/backend
creating build/temp.linux-x86_64-3.7/backend/built-in
creating build/temp.linux-x86_64-3.7/backend/built-in/src
creating build/temp.linux-x86_64-3.7/backend/custom
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c src/main.cpp -o build/temp.linux-x86_64-3.7/src/main.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c src/Config.cpp -o build/temp.linux-x86_64-3.7/src/Config.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/BaseBackend.cpp -o build/temp.linux-x86_64-3.7/backend/BaseBackend.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/BackendFactory.cpp -o build/temp.linux-x86_64-3.7/backend/BackendFactory.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/built-in/aclbackend.cpp -o build/temp.linux-x86_64-3.7/backend/built-in/aclbackend.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/built-in/src/model_process.cpp -o build/temp.linux-x86_64-3.7/backend/built-in/src/model_process.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/built-in/src/sample_process.cpp -o build/temp.linux-x86_64-3.7/backend/built-in/src/sample_process.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/built-in/src/utils.cpp -o build/temp.linux-x86_64-3.7/backend/built-in/src/utils.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DVERSION_INFO="1.0.2" -I/home/c00506053/dnmetis/backend_C++/dnmetis_backend/.eggs/pybind11-2.5.0-py3.7.egg/pybind11/include -I./inc/ -I./backend/inc -I./backend/built-in -I./backend/built-in/inc -I./backend/custom -I./backend/custom/inc -I/usr/local/Ascend/acllib/include/ -I/usr/local/python3.7.5/include/python3.7m -c backend/custom/trtbackend.cpp -o build/temp.linux-x86_64-3.7/backend/custom/trtbackend.o -w -O0 -fpermissive -std=c++14 -fvisibility=hidden
creating build/lib.linux-x86_64-3.7
g++ -pthread -shared build/temp.linux-x86_64-3.7/src/main.o build/temp.linux-x86_64-3.7/src/Config.o build/temp.linux-x86_64-3.7/backend/BaseBackend.o build/temp.linux-x86_64-3.7/backend/BackendFactory.o build/temp.linux-x86_64-3.7/backend/built-in/aclbackend.o build/temp.linux-x86_64-3.7/backend/built-in/src/model_process.o build/temp.linux-x86_64-3.7/backend/built-in/src/sample_process.o build/temp.linux-x86_64-3.7/backend/built-in/src/utils.o build/temp.linux-x86_64-3.7/backend/custom/trtbackend.o -L/usr/local/Ascend/acllib/lib64/ -L/usr/local/python3.7.5/lib -lascendcl -lpython3.7m -o build/lib.linux-x86_64-3.7/dnmetis_backend.cpython-37m-x86_64-linux-gnu.so -O0
creating build/bdist.linux-x86_64
creating build/bdist.linux-x86_64/egg
copying build/lib.linux-x86_64-3.7/dnmetis_backend.cpython-37m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
creating stub loader for dnmetis_backend.cpython-37m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/dnmetis_backend.py to dnmetis_backend.cpython-37.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying dnmetis_backend.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying dnmetis_backend.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying dnmetis_backend.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying dnmetis_backend.egg-info/not-zip-safe -> build/bdist.linux-x86_64/egg/EGG-INFO
copying dnmetis_backend.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
creating dist
creating 'dist/dnmetis_backend-1.0.2-py3.7-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing dnmetis_backend-1.0.2-py3.7-linux-x86_64.egg
removing '/usr/local/python3.7.5/lib/python3.7/site-packages/dnmetis_backend-1.0.2-py3.7-linux-x86_64.egg' (and everything under it)
creating /usr/local/python3.7.5/lib/python3.7/site-packages/dnmetis_backend-1.0.2-py3.7-linux-x86_64.egg
Extracting dnmetis_backend-1.0.2-py3.7-linux-x86_64.egg to /usr/local/python3.7.5/lib/python3.7/site-packages
dnmetis-backend 1.0.2 is already the active version in easy-install.pth

Installed /usr/local/python3.7.5/lib/python3.7/site-packages/dnmetis_backend-1.0.2-py3.7-linux-x86_64.egg
Processing dependencies for dnmetis-backend==1.0.2
Finished processing dependencies for dnmetis-backend==1.0.2


3、Check result of installation:\
"dnmetis-backend" will be seen
````pip3.7.5 list                                                                                             
Packag      Version
--------------- ---------
attrs           20.2.0
certifi         2020.6.20
cffi            1.14.2
chardet         3.0.4
decorator       4.4.2
dnmetis-backend 1.0.2
grpcio          1.31.0