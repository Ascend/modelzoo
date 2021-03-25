#accuracy:
python3.7 main.py --model=/data/hwwheel123/model/mobilenet-v3-large.om --image_size='224,224,3' --inputs='input:0' --outputs='MobilenetV3/Predictions/Softmax:0' --precision=fp16 --accuracy
#perf:
python3.7 main.py --model=/data/hwwheel123/model/mobilenet-v3-large.om --image_size='224,224,3' --inputs='input:0' --outputs='MobilenetV3/Predictions/Softmax:0' --precision=fp16




