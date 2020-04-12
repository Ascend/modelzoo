rm -rf dump*
rm -rf loss.log
rm -rf /var/log/npu/slog/*

pytest -s ./train_faster_rcnn_mindspore.py::test_faster_rcnn | tee log_fasterrcnn

rm -rf ge_graph
mkdir ge_graph
mv ge_* ge_graph/

rm -rf me_graph
mkdir me_graph
mv *.dot* me_graph/
mv *.dat* me_graph/
