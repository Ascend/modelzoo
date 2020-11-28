import xlrd 
import numpy as np
import sys
import math

def TotalTime(file):
	data = xlrd.open_workbook(file, encoding_override='utf-8')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	sum = 0
	max = 0
	min = 1000000
	count = 0.0
	for i in range(3, nrows):
		alldata = table.row_values(i)
		result = float(alldata[10])
		sum = sum + result		
		if max < result:
			max=result
		if min > result:
			min=result
	mean = round(sum / ( nrows - 3 ),2)	
	wave = round((max -min) / mean * 100,2)
	
	for i in range(3, nrows):
                alldata = table.row_values(i)
                result = float(alldata[10])
                if result > math.fabs(mean*1.05):
                	count = count + 1
	num = count / nrows*100
	print("total:",mean,max,min,wave,num,"%")
 
def Getnext(file):
	data = xlrd.open_workbook(file, encoding_override='utf-8')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	sum = 0
	max = 0
	min = 1000000
	for i in range(3, nrows):
		alldata = table.row_values(i)
		result = float(alldata[1])
		sum = sum + result		
		if max < result:
			max=result
		if min > result:
			min=result
	mean = round(sum / ( nrows - 3 ),2)	
	wave = round((max -min) / mean * 100,2)
	print("Getnext:",mean,max,min,wave,"%")
 
def FP_BP(file):
	data = xlrd.open_workbook(file, encoding_override='utf-8')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	sum = 0
	max = 0
	min = 1000000
	for i in range(3, nrows):
		alldata = table.row_values(i)
		result = float(alldata[2])
		sum = sum + result		
		if max < result:
			max=result
		if min > result:
			min=result
	mean = round(sum / ( nrows - 3 ),2)	
	wave = round((max -min) / mean * 100,2)
	print("FP_BP:",mean,max,min,wave,"%")
 
def bpend_iter(file):
	data = xlrd.open_workbook(file, encoding_override='utf-8')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	sum = 0
	max = 0
	min = 1000000
	for i in range(3, nrows):
		alldata = table.row_values(i)
		result = float(alldata[3])
		sum = sum + result		
		if max < result:
			max=result
		if min > result:
			min=result
	mean = round(sum / ( nrows - 3 ),2)	
	wave = round((max -min) / mean * 100,2)
	print("bpend_iter:",mean,max,min,wave,"%")
 

if __name__ == '__main__':
	input=sys.argv[1]
	TotalTime(input)
	Getnext(input)
	FP_BP(input)
	bpend_iter(input)

