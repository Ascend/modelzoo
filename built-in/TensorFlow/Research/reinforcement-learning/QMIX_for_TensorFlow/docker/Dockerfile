FROM ubuntu:18.04

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
COPY Huawei* /usr/local/share/ca-certificates/

COPY hosts /etc/hosts
COPY sources.list /etc/apt/sources.list
COPY pip.conf /root/.pip/
RUN apt-get update && apt-get install -y ca-certificates

RUN update-ca-certificates

ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu Packages
RUN apt-get update -y && apt-get install libhdf5-serial-dev python3-numpy python3-scipy libatlas-base-dev cmake gfortran -y && \
    apt-get install cmake \
	# qt5-default \
	libvtk6-dev zlib1g-dev \
	libjpeg-dev libwebp-dev \
	libpng-dev libtiff5-dev \
	libopenexr-dev \
        libgdal-dev libdc1394-22-dev \
	libavcodec-dev libavformat-dev \
	libswscale-dev -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache/* /tmp/*

RUN    apt-get update -y && apt-get install libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev \
                    libv4l-dev libxine2-dev libtbb-dev libeigen3-dev doxygen libgtk2.0-dev pkg-config -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache/* /tmp/*

# Ubuntu Packages
RUN apt-get update -y && \
    apt-get install -y apt-utils vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/* /root/.cache/* /tmp/* 

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git && \
    rm -rf /var/lib/apt/lists/*

# for debconf: unable to initialize frontend: Dialog 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /home/HwHiAiUser/.pip/
COPY pip.conf /home/HwHiAiUser/.pip/
#WORKDIR 
## end 


# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib imageio tensorboard-logger
#RUN python3 -m pip install numpy scipy pyyaml matplotlib imageio tensorboard-logger
RUN pip3 install pygame jsonpickle==0.9.6  setuptools

RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install


RUN mkdir /xt_install
WORKDIR /xt_install
## -- SMAC
# ModuleNotFoundError: No module named '_bz2'
RUN apt-get update -y && apt-get install libbz2-dev -y && rm -rf /var/lib/apt/lists/* /root/.cache/* /tmp/*
RUN ln -s /usr/lib/python3.6/lib-dynload/_bz2.cpython-36m-x86_64-linux-gnu.so /opt/buildtools/python3.7/lib/python3.7/site-packages/_bz2.so

#RUN wget -c http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz  && tar -zxvf bzip2-1.0.6.tar.gz && cd bzip2-1.0.6 && \
#	mkdir build && cd build &&  make .. && make install
RUN pip3 install --upgrade pip==20.0.2

# could used SMAC-0.1.0b1 version 
RUN pip3 install git+https://github.com/oxwhirl/smac.git
#ENV SC2PATH /pymarl/3rdparty/StarCraftII
ENV SC2PATH /pymarl-sim/StarCraftII

## xingtian 
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y libopencv-dev && \
    apt-get install -y redis-server python3-tk && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools


####

RUN pip3 install h5py pybind11 scipy 


COPY src_zip/opencv-4.2.0.zip /xt_install
RUN unzip opencv-4.2.0.zip && cd opencv-4.2.0 && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/opencv .. && \
    make -j32 && make install && \
    cd ../../ &&  rm opencv-4.2.0.zip opencv-4.2.0 -rf

RUN sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
RUN sh -c 'echo "/usr/local/opencv/lib" >> /etc/ld.so.conf.d/opencv.conf'

#RUN ln -s /usr/local/opencv/lib/python3.7/site-packages/cv2/python-3.7/cv2.cpython-37m-aarch64-linux-gnu.so /usr/local/lib/python3.7/site-packages/cv2.so
RUN ln -s /usr/local/opencv/lib/python3.7/site-packages/cv2/python-3.7/cv2.cpython-37m-aarch64-linux-gnu.so /opt/buildtools/python3.7/lib/python3.7/site-packages/cv2.so

RUN pip3 install lz4 psutil tqdm pyzmq gym atari-py redis pyyaml fabric2

RUN pip3 install zmq imageio matplotlib==3.0.3 Ipython tensorboardX

# for profiling 
RUN pip3 install xlsxwriter xlrd tabulate openpyxl pandas

COPY src_zip/smac.tar /xt_install
RUN tar xvf smac.tar && cd smac && pip3 install ./  && cd ../
# Successfully installed SMAC-0.1.0b1
# NOTE: now, smac github update to 1.0.0, we used SMAC-0.1.0b1


