 **Introduce** 

Envperformancecheck tool is used to scan the hardware resources of the training environment, make a general check on the main specifications and parameters of the environment (operating system, CPU model, usage statistics, average load, memory capacity, disk resources and device, etc.), and give the check result report. For abnormal parameter items, they are displayed in red

 **File structure** 

EnvCheck.sh

 **Instructions** 

======Operation mode: bash envcheck.sh/npu

======Parameter Description: / NPU training dataset archive directory, used to check the size of partition directory. If it is not specified, it is / NPU by default

 **Matters needing attention** 

For the check of computing power, you need to obtain the npusmi tool query of product release

Supplement some requirements for relevant parameters obtained by npusmi tool:

Device count no less than 8
Device health status should be 0 (normal)
Device memory frequency no less than 1000 HZ
Device HBM frequency no less than 1000 HZ
RoCE network health status should be 0(RDFX_DETECT_OK) or 6(RDFX_DETECT_INIT)
AICPU current frequency no less than 2000 
AICPU num no less than 14
AICore frequency no less than 1000 HZ
