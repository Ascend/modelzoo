import os
import sys
import subprocess

if __name__ =="__main__":
    model = sys.argv[1]
    input = sys.argv[2]
    output = sys.argv[3]
    label = sys.argv[4]
    path = os.path.realpath(__file__)
    path = path.rsplit('/',1)[0]
    inference = subprocess.Popen('./msame --model {} --input {} --output {}'.format(model,input,output),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True) 
    inference = inference.stdout.readlines()
    avg_time = 0
    for line in inference:
        line = line.decode()
        print(line.replace('\n',''))
        if "Inference average time without first time" in line:
            avg_time = float(line.split("Inference average time without first time:")[1].split("ms")[0])
    
    postprocess = subprocess.Popen('python3.7.5 accuarcy_top1.py {} {}'.format(output,label),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True)
    postprocess = postprocess.stdout.readlines()
    top1 = 0
    for line in postprocess:
        line = line.decode()
        print(line.replace('\n',''))
        if "Top1 accuarcy:" in line:
            top1 = float(line.split("Top1 accuarcy:")[1].split("/n")[0])
    
    expect_time = 145
    expect_top1 = 0.86
    print("Average inference time is %.2f ms, expect time is <%.2f ms"%(avg_time,expect_time))
    print("Top1 accuarcy is %.2f, expect top1 is >%.2f"%(top1,expect_top1))
    if avg_time < expect_time and top1 > expect_top1:
        print('Run testcase success!')
    else:
        print('Run testcase failed!')



