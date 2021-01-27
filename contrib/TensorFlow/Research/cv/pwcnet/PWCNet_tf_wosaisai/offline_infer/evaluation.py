import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='', 
                    help='output path.')
args = parser.parse_args()

if __name__ == '__main__':
    gt = args.gt_path
    prediction = args.output_path

    imgs = os.listdir(gt)
    preds = os.listdir(prediction)
    imgs = sorted(imgs)
    preds = sorted(preds)

    res = []

    for img, pred in zip(imgs, preds):
        label = os.path.join(gt, img)
        out = os.path.join(prediction, pred)

        label = np.fromfile(label, dtype=np.float32)
        out = np.fromfile(out, dtype=np.float32)

        label = label.reshape(448, 1024, 2)
        out = out.reshape(448, 1024, 2)
        label = label[:436, :, :]
        out = out[:436, :, :]
    
        res.append(np.mean(np.linalg.norm(label - out, axis=-1)))

    print('Average EPE = ', np.mean(res))
