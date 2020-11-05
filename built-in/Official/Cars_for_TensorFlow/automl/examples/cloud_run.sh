pip install pareto==1.1.1.post3
pip install distributed==2.12.0 PyYAML==5.1.2
pip install MarkupSafe Jinja2 packaging typing_extensions locket  partd fsspec
pip install bokeh
pip install dask[complete]
pip install py-dag==3.0.1
pip install mmcv==0.2.14
pip install tf-models-official==0.0.3.dev1
pip install prettytable==0.7.2
#pip install scikit-learn==0.22.1

cp -rf /home/work/user-job-dir/automl  /cache

export PATH=/usr/local/ma/python3.7/bin:$PATH

export PYTHONPATH=/cache/automl:$PYTHONPATH

cd /cache/workspace/device0

#export DUMP_GE_GRAPH=1
#export EXPERIMENTAL_DYNAMIC_PARTITION=1

#export PROFILING_DIR=/cache/workspace/device0/profiling/container/0
#export FP_POINT=resnet_v1_50_1/conv1/Conv2D
#export BP_POINT=add_1

python /cache/automl/examples/run_example_roma.py $1 bj4_y