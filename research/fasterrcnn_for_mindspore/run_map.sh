source ./env.sh

rm -rf run_map
cp ./train_faster_rcnn ./run_map -rf
cd run_map

pytest -s train_faster_rcnn_mindspore.py::test_faster_rcnn_mAP | tee log_map 2>&1

cd ..
