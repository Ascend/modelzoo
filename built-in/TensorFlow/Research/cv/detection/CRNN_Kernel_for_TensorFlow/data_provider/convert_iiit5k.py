from scipy import io
import numpy as np 
import argparse 

###############################################
# load testdata 
# testdata.mat structure 
# test[:][0] : image name 
# test[:][1] : label 
# test[:][2] : 50 lexicon
# test[:][3] : 1000 lexicon 
##############################################

def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-m', '--mat_file', type=str, default='testdata.mat',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--output_dir', type=str, default='./processed',
                        help='Directory where ord map dictionaries for the dataset were stored')

    parser.add_argument('-a', '--output_annotation', type=str, default='./annotation.txt',
                        help='Directory where ord map dictionaries for the dataset were stored')

    return parser.parse_args()




def mat_to_list(mat_file):
    ann_ori = io.loadmat(mat_file)
    testdata = ann_ori['testdata'][0]
    
    ann_output = []
    for elem in testdata:
        img_name = elem[0][0]
        label = elem[1][0]
        print('image name ', img_name, 'label: ', label)
        ann = img_name+',' + label
        ann_output.append(ann)
    return ann_output

def convert():

    
    args = init_args()

    ann_list = mat_to_list(args.mat_file)

    print("output ann : ",args.output_annotation)
    ann_file = args.output_annotation
    with open(ann_file, 'w') as f:
        for line in ann_list:
            txt = line + '\n'
            f.write(txt)



if __name__ == "__main__":
    convert()



#path = "/data/m00536736/modelzoo/OCR/datasets/IIIT5K/testdata.mat"
#testfile = io.loadmat(path)
#testdata = testfile['testdata'][0]
#
#for i in range(100):
#    tmp = testdata[i]
#    print("image name :", tmp[0])
#    print("image label : ", tmp[1])
#    print("first 5 word from 50  lexicon : ",tmp[2][:5])
#    print("first 5 word from 1k  lexicon : ",tmp[3][:5])
#
#
#np.save('/data/m00536736/modelzoo/OCR/crnn_smoke/data/test/iiit5k/testdata.npy',testdata)
#




