import os

ch2_gt_dir="./ch2_gt/"
ch2_gt_new_dir="./ch2_gt_new/"

if __name__ == "__main__":
    gt_files = os.listdir(ch2_gt_dir)
    gt_files.sort()
    for file in gt_files:
        if file.endswith(".txt"):
            new_gt = ""
            f = open(os.path.join(ch2_gt_dir, file))
            lines = f.readlines()
            for line in lines:
                if line:
                    tmp = line.split(" ")
                    xmin = tmp[0]
                    ymin = tmp[1]
                    xmax = tmp[2]
                    ymax = tmp[3]
                    content = tmp[4][1:-2]
                    new_gt += '{},{},{},{},{},{},{},{},{}\r\n'.format(xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,content)
            f.close()
            with open(os.path.join(ch2_gt_new_dir, file), 'w') as f:
                f.write(new_gt)
            f.close()
