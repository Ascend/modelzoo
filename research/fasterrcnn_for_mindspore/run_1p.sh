source ./env.sh

rm -rf run_single
cp ./train_faster_rcnn ./run_single -rf
cd run_single

pytest -s ../train_faster_rcnn/train_faster_rcnn_mindspore.py::test_train_faster_rcnn | tee log_run_single

cd ..
