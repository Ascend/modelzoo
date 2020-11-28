import csv
import sys


net_name = sys.argv[1]
file_name = sys.argv[2]
rank_size = float(sys.argv[3])
batch_size = float(sys.argv[4])

list_total = []
with open(file_name,'r+') as file:
    reader = csv.reader(file)
    for i in reader:
        if net_name in i:
            list_total = list(i)
# print(list_total[-1])
step_time = float(list_total[-1])
FPS = (1000.0/step_time)*rank_size*batch_size
print("avg_total_time =", float(list_total[-1]))
print("FPS =", int(FPS + 0.5))