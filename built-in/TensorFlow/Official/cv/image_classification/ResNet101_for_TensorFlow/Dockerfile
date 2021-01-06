ARG FROM_IMAGE_NAME=ascend-tensorflow-arm:20.1.0
FROM ${FROM_IMAGE_NAME}

RUN apt install git

COPY requirements.txt .
RUN pip3.7 install -r requirements.txt