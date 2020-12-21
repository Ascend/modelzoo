ARG FROM_IMAGE_NAME=ascend-pytorch-arm:20.1.0
FROM ${FROM_IMAGE_NAME}

COPY requirements.txt .
RUN pip3.7 install -r requirements.txt
RUN apt-get install libopenblas-dev liblapack-dev libatlas-base-dev && \
        sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev && \
        sudo apt-get install git cmake build-essential libjpeg62-dev
RUN wget https://github.com/opencv/opencv/archive/3.4.12.zip --no-check-certificate && \
        unzip 3.4.12.zip && \
        cd ./opencv-3.4.12 && \
        cmake -D WITH_IPP=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF -D CMAKE_BUILD_TYPE=RELEASE \
        -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF \
        -D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7 \
        -D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m/ \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include \
        -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages \
        -D PYTHON_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so.1.0 ..

