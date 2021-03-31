import argparse 


path="/data/m00536736/modelzoo/OCR/datasets/2013/Challenge2_Test_Task3_GT.txt"
ann_file = open(path,'r')

annotation_list = [line.strip("\n") for line in ann_file.readlines()]


def is_valid_char(ch):
    ch_ord = ord(ch)
    
    ord_0 = ord('0')
    ord_9 = ord('9')
    ord_a = ord('a')
    ord_z = ord('z')

    if (ch_ord>=ord_0 and ch_ord<=ord_9) or (ch_ord>=ord_a and ch_ord<=ord_z):
        return True
    else:
        return False

def get_abnormal_list(ann_list):
    abn_list = []
    for ann in ann_list:
        label = ann.split(',')[1]
        label = label.strip().lower()
        for l in label:
            flag = is_valid_char(l)
            if not flag:
                abn_list.append(ann)
                print(ann)
                break
    print("number of abnormal annotation :", len(abn_list))
    return abn_list



path="/data/m00536736/modelzoo/OCR/datasets/2013/Challenge2_Test_Task3_GT.txt"
ann_file = open(path,'r')

annotation_list = [line.strip("\n") for line in ann_file.readlines()]
ann_file.close()

abn_list = get_abnormal_list(annotation_list)

path="/data/m00536736/modelzoo/OCR/datasets/2013/processed_annotation.txt"

clean_list = [line for line in annotation_list if line not in abn_list]

with open(path,'w') as f:
    for line in clean_list:
        line = line +'\n'
        f.write(line)
