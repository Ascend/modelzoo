import os
import numpy as np
import sys
from multiprocessing import Process
import time
import subprocess

def bin2float(file,dtype):
    if file.endswith(".bin"):
        if dtype == "fp32":
            data = np.fromfile(file,dtype='float32')
        elif dtype == "fp16":
            data = np.fromfile(file,dtype='float16')
        elif dtype == "int32":
            data = np.fromfile(file,dtype=np.int32)
        elif dtype == "int8":
            data = np.fromfile(file,dtype=np.int8)
        else:
            print("Input dtype error!")
            return 0
        float_file=file+".txt"
        print("save the file: "+float_file)
        np.savetxt(float_file,data.reshape(-1,1),fmt='%.6f')

def bintofloat(filename,dtype):
    if os.path.isdir(filename):
        process= []
        i=0
        for file in os.listdir(filename):
            if(file != "." and file !=".."):
                process.append(Process(target=bin2float, args=(filename+file,dtype,)))
                process[i].start()
                i +=1
    else:
        bin2float(filename,dtype)

if __name__ == "__main__":
    subprocess.run("ulimit -n 65535",shell=True,cwd="./")
    print("params: " + sys.argv[1] + "," + sys.argv[2])
    bintofloat(sys.argv[1],sys.argv[2])

