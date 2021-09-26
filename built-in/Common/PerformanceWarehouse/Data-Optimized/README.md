## Introduce

enable_data_pre_proc_scan.py is a tool used to scan the Tensorflow Python script whether can be enabled the TDT High-speed channel,
To use this tool,you first need to train with environment variable by setting the following script:
```
export PRINT_MODEL=1
```
After training, you will see the BeforeSubGraph_*.pbtxt in the code path.

## File structure

enable_data_pre_proc.py

## Instructions

- Operation mode: 
  
  python3 enable_data_pre_proc_scan.py --scan_path=param1


- Parameter Description: 
  
  param1 means the path of Tensorflow Python script which store the BeforeSubGraph_*.pbtxt file. 
If it is not specified, this tool will raise a error.


- Result: 
  
  If the print info: "**Dataset exists and enable_data_pre_proc is false:**" contains your model script directory,
  then you can set enable_data_pre_proc to use TDT High-speed channel.
  
  About how to set enable_data_pre_proc please see details:
  [TensorFlow model migration guide](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtg_13_9001.html)


## Matters needing attention

This tool is only used for Tensorflow not for Pytorch and other frame.
