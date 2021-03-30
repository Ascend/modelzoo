 ps -ef | grep TdtMain | awk '{print $2}' | xargs kill -9
rm -rf *.pbtxt
rm -rf /var/log/npu/slog/*.log
rm ckpt* -rf
find ./ -name "*.pyc" | xargs rm -rf
find ./ -name __pycache__ | xargs rm -rf
rm /var/log/npu/dataset/* -rf
