#accuracy:
python3.7 main.py --model=/data/hwwheel123/model/xception_tf.om --image_size='299,299,3' --inputs='input_1:0' --outputs='dense_1/Softmax:0' --precision=fp16 --accuracy

#perf:
python3.7 main.py --model=/data/hwwheel123/model/xception_tf.om --image_size='299,299,3' --inputs='input_1:0' --outputs='dense_1/Softmax:0' --precision=fp16





