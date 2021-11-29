ARG FROM_IMAGE_NAME=ascend-tensorflow-arm:20.1.0
FROM ${FROM_IMAGE_NAME}

RUN apt -y install libgl1-mesa-glx

COPY requirements.txt .
RUN pip3.7 install -r requirements.txt

RUN ln -s /usr/bin/python3.7 /usr/bin/python3