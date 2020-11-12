import os
import numpy as np
from keras.preprocessing.image import img_to_array

txt_path= "./ground_truth/public_test.txt"
file_path= "./input/"

with open(txt_path, 'r') as f:
     reader = f.readlines()
     i = 0
     for row in reader:
         if "PublicTest" in row:
             print("start to preprocess image file: %d"%i)
             data = row.split(",")[2].split(" ")
             data = list(map(int,data))
             img = np.array(data).reshape(48,48)
             img = img.astype(np.float) / 255.
             img = img_to_array(img)
             tmp = np.expand_dims(img, 0)
             tmp.tofile(file_path+"/"+str(i.__str__().zfill(6))+".bin")
             i+=1
