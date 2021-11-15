ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG SDK_PKG

RUN ln -s  /usr/local/python3.7.5/bin/python3.7 /usr/bin/python

RUN apt-get update && \
    apt-get install libglib2.0-dev -y || \
    rm -rf /var/lib/dpkg/info && \
    mkdir /var/lib/dpkg/info && \
    apt-get install libglib2.0-dev -y && \
    pip install pytest-runner && \
    pip install mmcv
    