ARG FROM_IMAGE_NAME=ascend-tensorflow-arm:20.1.0
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y libsndfile1-dev libjpeg-turbo8-dev  pbzip2

COPY requirements.txt .
RUN pip3.7 install -r requirements.txt