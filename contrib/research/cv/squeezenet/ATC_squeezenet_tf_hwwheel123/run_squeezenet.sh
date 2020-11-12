#accuracy:
python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy
#perf:
python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 




