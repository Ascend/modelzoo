import numpy as np
import os


if __name__ == '__main__':
    gt = './gt'
    prediction = './prediction'

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

        res.append(np.mean(np.pow(label - out, 2)))

    print(np.mean(res))
