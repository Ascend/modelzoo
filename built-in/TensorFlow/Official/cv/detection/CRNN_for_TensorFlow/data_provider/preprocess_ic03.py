import argparse 



def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-s', '--src_ann', type=str, default='annotation.txt',
                        help='path to original annotation ')
    parser.add_argument('-o', '--dst_ann', type=str, default='processed_annotation.txt',
                        help='path to filtered annotation')


    return parser.parse_args()



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
    
        if len(label)<3:
            abn_list.append(ann)
            continue

        for l in label:
            flag = is_valid_char(l)
            if not flag:
                abn_list.append(ann)
                #print(ann)
                break
    print("number of abnormal annotation :", len(abn_list))
    return abn_list



def filter():

    args = init_args()

    ann_file = open(args.src_ann,'r')
    annotation_list = [line.strip("\n") for line in ann_file.readlines()]
    ann_file.close()
    
    abn_list = get_abnormal_list(annotation_list)
    clean_list = [line for line in annotation_list if line not in abn_list]
    print("number of annotation after filtering :{}".format(len(clean_list)))

    output = args.dst_ann
    with open(output,'w') as f:
        for line in clean_list:
            line = line +'\n'
            f.write(line)




if __name__=="__main__":
    filter()


#path="/data/m00536736/modelzoo/OCR/datasets/2003/SceneTrialTest/annotation.txt"
#ann_file = open(path,'r')
#
#annotation_list = [line.strip("\n") for line in ann_file.readlines()]
#ann_file.close()
#
#abn_list = get_abnormal_list(annotation_list)
#
#path="/data/m00536736/modelzoo/OCR/datasets/2003/SceneTrialTest/processed_annotation.txt"
#
#clean_list = [line for line in annotation_list if line not in abn_list]
#print("number of annotation after filtering :{}".format(len(clean_list)))


#with open(path,'w') as f:
#    for line in clean_list:
#        line = line +'\n'
#        f.write(line)
