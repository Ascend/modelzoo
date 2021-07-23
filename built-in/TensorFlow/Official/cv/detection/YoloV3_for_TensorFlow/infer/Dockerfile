ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG SDK_PKG

RUN ln -s  /usr/local/python3.7.5/bin/python3.7 /usr/bin/python
RUN ln -s  /usr/local/python3.7.5/bin/python3.7 /usr/bin/python3

RUN apt-get update && \
    apt -y install libgl1-mesa-glx

COPY requirements.txt .
RUN pip3.7 install -r requirements.txt

# pip install sdk_run
COPY $SDK_PKG .
RUN ls -hrlt
RUN chmod +x ${SDK_PKG} && \
    ./${SDK_PKG}  --install-path=/home/run --install && \
     bash -c "source ~/.bashrc"