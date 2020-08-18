import os
import sys


def check_performance(steps,loop,path):
    #print("##########################FPS_RESULT############################")
    item_list=[]
    try:
        file_FPS = open(path+"/FPS.log")
        List_FPS = []
        sum = 0
        for line in file_FPS:
            number = line.split("\n")
            FPS = float(number[0])
            List_FPS.append(FPS)
        file_FPS.close()
        del(List_FPS[0:2])
        num = len(List_FPS)
        for i in range(num):
            sum += float(List_FPS[i])
        average =round( sum / num)
        max_FPS =round( max(List_FPS),2)
        min_FPS = round(min(List_FPS),2)
        fluctuation = round(((float(max_FPS) - float(min_FPS))/average)*1000)
        max_train_steps=steps
        iterations_per_loop=loop
        num=0
        for value in List_FPS:
             if float(value) < average:
                num+=1
        Proportion = float(num)/(float(max_train_steps)/float(iterations_per_loop))
        print('average:%s,max_FPS:%s,min_FPS:%s,fluctuation:%s,Proportion:%s'%(average,max_FPS,min_FPS,fluctuation,Proportion))
    except Exception as e:
        print("list index out of reange")
	#print(item_list)
    #return item_list



if __name__ == '__main__':
    steps = sys.argv[1]
    loop = sys.argv [2]
    path = sys.argv [3]
    check_performance(steps,loop,path)
