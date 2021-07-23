# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

detection_path = "detections/"
all_detections = os.listdir(detection_path)

cnter = 0
for detec in all_detections:
    if cnter%100==0:
        print("---> PROGRESS: %i" %cnter)
    # print("CURRENT RESULT: "+detec)
    newContent = []
    with open(detection_path+detec) as f:
        content = f.readlines()
        for line in content:
            linePart1 = line.split(" ")[0]
            linePart2 = line.split(" ")[1:]
            newLine = linePart1+" 1 "+linePart2[0]+" "+linePart2[1]+" "+linePart2[2]+" "+linePart2[3]
            newContent.append(newLine)
    file_to_be_write = open(detection_path+detec, "w")
    for line in newContent:
        file_to_be_write.write(line)
    file_to_be_write.close
    cnter+=1

print("EVERY FILE HAS BEEN PROCESSED: %i" %cnter)
# print(all_detections)