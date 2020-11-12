import argparse
import numpy as np

def generate_random_data(file_path,image_num=100,height=224,width=224):
    print("height: %d"%height)
    print("width: %d"%width)
    for i in range(image_num):
        input_data = np.random.randn(1,3,height,width).astype(np.float32)
        input_data.tofile(file_path+"/"+str(i.__str__().zfill(6))+".bin")
    print("num:%d random datas has been created in path:%s" %(image_num,file_path))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p",type=str,default="./")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--h", type=int, default=224)
    parser.add_argument("--w", type=int, default=224)
    args = parser.parse_args()

    generate_random_data(args.p,args.n,args.h,args.w)

