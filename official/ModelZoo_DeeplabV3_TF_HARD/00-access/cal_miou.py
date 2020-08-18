


import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_log', default="",
                         help="""result log file.""")
args = parser.parse()

with open(args.result_log, 'r') as f:

    lines = f.readlines()
    eval_res = [l.strip('\n') for l in lines if l.startswith('eval/miou')]

    num_ckpt = len(eval_res)//21
    # print("{} checkpoints existed ".format(num_ckpt))
    for i in range(0,num_ckpt):
        miou_tot = 0
        for j in range(0,21):
            index = i * 21 + j
            iou_val = float(eval_res[index].split(':')[-1].strip())

            miou_tot += iou_val
        miou = miou_tot/21.0
        print("mean IOU is :",str(miou))




    
