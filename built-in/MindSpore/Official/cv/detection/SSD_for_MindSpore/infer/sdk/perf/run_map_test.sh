#!/bin/bash

PY=/usr/bin/python3.7

export PYTHONPATH=${PYTHONPATH}:.

${PY} generate_map_report.py \
--annotations_json=/home/dataset/coco2017/annotations/instances_val2017.json \
--det_result_json=/home/sam/codes/SSD_MobileNet_FPN_for_MindSpore/infer/sdk/perf/om_infer_output_on_coco_val2017/om_det_result.json \
--output_path_name=./map_output/map.txt \
--anno_type=bbox