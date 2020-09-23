import sys
import xlrd
import numpy as np
import random
def operating_execl(loop,path):

    path = path
#    num = random.randint(0,7)
   # print(num)
    num = 0
    filename = path + "analysis_performance_tag." + str(num) + ".xlsx"
    count = 0

    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]

    ncols = table.ncols
    #print(ncols) #you xiao lie 17 cong 0 kai shi
    #print(table.col(1,start_rowx = 0,end_rowx = 5))
    nrows = table.nrows
    #print(nrows) #you xiao hang 10001
    liter_end_to_next_FP_start_loop = []
    liter_end_to_next_FP_start_remove_loop = []
    BP_and_FP_loop = []
    BP_and_FP_remove_loop = []
    BP_end_to_iter_end_loop = []
    BP_end_to_iter_end_remove_loop = []
    total_time_loop = []
    total_time_remove_loop = []

    for i in range(int(nrows)):
        if (i-1) % int(loop) == 0 or i == 0:
            liter_end_to_next_FP_start_loop.append(table.col_values(1,start_rowx = i,end_rowx = i+1)[0])
            BP_and_FP_loop.append(table.col_values(2,start_rowx = i,end_rowx = i+1)[0])
            BP_end_to_iter_end_loop.append(table.col_values(3,start_rowx = i,end_rowx = i+1)[0])
            total_time_loop.append(table.col_values(10,start_rowx = i,end_rowx = i+1)[0])
        else:
            liter_end_to_next_FP_start_remove_loop.append(table.col_values(1,start_rowx = i,end_rowx = i+1)[0])
            BP_and_FP_remove_loop.append(table.col_values(2, start_rowx=i, end_rowx=i + 1)[0])
            BP_end_to_iter_end_remove_loop.append(table.col_values(3, start_rowx=i, end_rowx=i + 1)[0])
            total_time_remove_loop.append(table.col_values(10, start_rowx=i, end_rowx=i + 1)[0])

    for o in range(2):
        del liter_end_to_next_FP_start_loop[0]
        del BP_and_FP_loop[0]
        del BP_end_to_iter_end_loop[0]
        del total_time_loop[0]


    # print(list_liter_end_to_next_FP_start_remove_loop)
    # print(list_liter_end_to_next_FP_start_loop)
    #max value in loop
    max_liter_end_to_next_FP_start_loop = max(liter_end_to_next_FP_start_loop)
    max_BP_and_FP_loop = max(BP_and_FP_loop)
    max_BP_end_to_iter_end_loop = max(BP_end_to_iter_end_loop)
    max_total_time_loop = max(total_time_loop)
    # print("loop点各项最大值：迭代间隙：%s，正反向：%s，汇聚拖尾：%s，allreduce1：%s，allreduce2：%s，端到端：%s"%(max_liter_end_to_next_FP_start_loop,max_BP_and_FP_loop,max_BP_end_to_iter_end_loop,max_AllReduce1_total_loop,max_AllReduce2_total_loop,max_total_time_loop))


    #min value in loop
    min_liter_end_to_next_FP_start_loop = min(liter_end_to_next_FP_start_loop)
    min_BP_and_FP_loop = min(BP_and_FP_loop)
    min_BP_end_to_iter_end_loop = min(BP_end_to_iter_end_loop)
    min_total_time_loop = min(total_time_loop)
    # print("loop点各项最小值：迭代间隙：%s，正反向：%s，汇聚拖尾：%s，allreduce1：%s，allreduce2：%s，端到端：%s"%(min_liter_end_to_next_FP_start_loop,min_BP_and_FP_loop,min_BP_end_to_iter_end_loop,min_AllReduce1_total_loop,min_AllReduce2_total_loop,min_total_time_loop))

    #max value not in loop
    max_liter_end_to_next_FP_start_remove_loop = max(liter_end_to_next_FP_start_remove_loop)
    max_BP_and_FP_remove_loop = max(BP_and_FP_remove_loop)
    max_BP_end_to_iter_end_remove_loop = max(BP_end_to_iter_end_remove_loop)
    max_total_time_remove_loop = max(total_time_remove_loop)

    # print("去掉loop之后各项最大值：迭代间隙：%s，正反向：%s，汇聚拖尾：%s，allreduce1：%s，allreduce2：%s，端到端：%s"%(max_liter_end_to_next_FP_start_remove_loop,max_BP_and_FP_remove_loop,max_BP_end_to_iter_end_remove_loop,max_AllReduce1_total_remove_loop,max_AllReduce2_total_remove_loop,max_total_time_remove_loop))
    #min value not in loop
    min_liter_end_to_next_FP_start_remove_loop = min(liter_end_to_next_FP_start_remove_loop)
    min_BP_and_FP_remove_loop = min(BP_and_FP_remove_loop)
    min_BP_end_to_iter_end_remove_loop = min(BP_end_to_iter_end_remove_loop)
    min_total_time_remove_loop = min(total_time_remove_loop)
    #print(len(list_liter_end_to_next_FP_start_loop))
    # print("去掉loop之后各项最小值：迭代间隙：%s，正反向：%s，汇聚拖尾：%s，allreduce1：%s，allreduce2：%s，端到端：%s"%(min_liter_end_to_next_FP_start_remove_loop,min_BP_and_FP_remove_loop,min_BP_end_to_iter_end_remove_loop,min_AllReduce1_total_remove_loop,min_AllReduce2_total_remove_loop,min_total_time_remove_loop))

    #average not in loop
    average_liter_end_to_next_FP_start_remove_loop = np.mean(liter_end_to_next_FP_start_remove_loop)
    average_BP_and_FP_remove_loop = np.mean(BP_and_FP_remove_loop)
    average_BP_end_to_iter_end_remove_loop = np.mean(BP_end_to_iter_end_remove_loop)
    average_total_time_remove_loop = np.mean(total_time_remove_loop)
    # print("去掉loop之后各项均值：迭代间隙：%s，正反向：%s，汇聚拖尾：%s，allreduce1：%s，allreduce2：%s，端到端：%s"%(average_liter_end_to_next_FP_start_remove_loop,average_BP_and_FP_remove_loop,average_BP_end_to_iter_end_remove_loop,average_AllReduce1_total_remove_loop,average_AllReduce2_total_remove_loop,average_total_time_remove_loop))

    #average in loop
    average_liter_end_to_next_FP_start_loop = np.mean(liter_end_to_next_FP_start_loop)
    average_BP_and_FP_loop = np.mean(BP_and_FP_loop)
    average_BP_end_to_iter_end_loop = np.mean(BP_end_to_iter_end_loop)
    average_total_time_loop = np.mean(total_time_loop)
    # print("loop点各项均值：迭代间隙：%s，正反向：%s，汇聚拖尾：%s，allreduce1：%s，allreduce2：%s，端到端：%s"%(average_liter_end_to_next_FP_start_loop,average_BP_and_FP_loop,average_BP_end_to_iter_end_loop,average_AllReduce1_total_loop,average_AllReduce2_total_loop,average_total_time_loop))

    wave_total = (max_total_time_remove_loop - min_total_time_remove_loop) / average_total_time_remove_loop * 100
    wave_Getnext = (max_liter_end_to_next_FP_start_remove_loop - min_liter_end_to_next_FP_start_remove_loop) / average_liter_end_to_next_FP_start_remove_loop * 100
    wave_FP_BP = (max_BP_and_FP_remove_loop - min_BP_and_FP_remove_loop) / average_BP_and_FP_remove_loop * 100
    wave_bpend_iter = (max_BP_end_to_iter_end_remove_loop - min_BP_end_to_iter_end_remove_loop) / average_BP_end_to_iter_end_remove_loop * 100

    for total_time in total_time_remove_loop:
        if total_time >  average_total_time_remove_loop * 1.05:
            count += 1
    num_1 = count / len(total_time_remove_loop) * 100



    print("total: ",average_total_time_remove_loop,max_total_time_remove_loop,min_total_time_remove_loop,wave_total,num_1,"%")
    print("Getnext: ",average_liter_end_to_next_FP_start_remove_loop,max_liter_end_to_next_FP_start_remove_loop,min_liter_end_to_next_FP_start_remove_loop,wave_Getnext,"%")
    print("FP_BP: ",average_BP_and_FP_remove_loop,max_BP_and_FP_remove_loop,min_BP_and_FP_remove_loop,wave_FP_BP,"%")
    print("bpend_iter: ",average_BP_end_to_iter_end_remove_loop,max_BP_end_to_iter_end_remove_loop,min_BP_end_to_iter_end_remove_loop,wave_bpend_iter,"%")


if __name__ == "__main__":
    loop = sys.argv[1]
    path = sys.argv[2]
    operating_execl(loop,path)

