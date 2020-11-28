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
	mean = sum / ( nrows - 3 )	
	wave = (max -min) / mean * 100
	
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
	mean = sum / ( nrows - 3 )	
	wave = (max -min) / mean * 100
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
	mean = sum / ( nrows - 3 )	
	wave = (max -min) / mean * 100
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
	mean = sum / ( nrows - 3 )	
	wave = (max -min) / mean * 100
	print("bpend_iter:",mean,max,min,wave,"%")
 
def AR1(file):
	data = xlrd.open_workbook(file, encoding_override='utf-8')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	sum = 0
	max = 0
	min = 1000000
	for i in range(3, nrows):
		alldata = table.row_values(i)
		result = float(alldata[5])
		sum = sum + result		
		if max < result:
			max=result
		if min > result:
			min=result
	mean = sum / ( nrows - 3 )	
	wave = (max -min) / mean * 100
	print("AR1:",mean,max,min,wave,"%")


def AR2(file):
	data = xlrd.open_workbook(file, encoding_override='utf-8')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	sum = 0
	max = 0
	min = 1000000
	for i in range(3, nrows):
		alldata = table.row_values(i)
		result = float(alldata[8])
		sum = sum + result		
		if max < result:
			max=result
		if min > result:
			min=result
	mean = sum / ( nrows - 3 )	
	wave = (max -min) / mean * 100
	print("AR2:",mean,max,min,wave,"%")

if __name__ == '__main__':
	input=sys.argv[1]
	TotalTime(input)
	Getnext(input)
	FP_BP(input)
	bpend_iter(input)
	AR1(input)
	AR2(input)

