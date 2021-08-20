import os
import json
from tqdm import tqdm

with open('./instances_val2017.json', 'r') as file:
    content = file.read()
content = json.loads(content)
info = content.get('info')
licenses = content.get('licenses')
images = content.get('images')
annotations = content.get('annotations')
categroies = content.get('categories')

with open('./coco2017.names', 'w') as f:
    for categroie in categroies:
        categroie_name = categroie.get('name')
        if len(categroie_name.split()) == 2:
            temp = categroie_name.split()
            categroie_name = temp[0] + '_' + temp[1]
        f.write(categroie_name)
        f.write('\n')

file_names = [image.get('file_name') for image in images]
widths = [image.get('width') for image in images]
heights = [image.get('height') for image in images]
image_ids = [image.get('id') for image in images]
assert len(file_names) == len(widths) == len(heights) == len(image_ids), "must be equal"

annotation_ids = [annotation.get('image_id') for annotation in annotations]
bboxs = [annotation.get('bbox') for annotation in annotations]
category_ids = [annotation.get('category_id') for annotation in annotations]
segmentations = [annotation.get('segmentation') for annotation in annotations]
iscrowds = [annotation.get('iscrowd') for annotation in annotations]

assert len(annotation_ids) == len(bboxs) == len(category_ids) ==len(segmentations) # 255094

with open('coco_2017.info', 'w') as f:
    for index, file_name in enumerate(file_names):
        file_name = 'val2017/' + file_name
        line = "{} {} {} {}".format(index, file_name, widths[index], heights[index])
        f.write(line)
        f.write('\n')


def get_all_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]


def get_categroie_name(lst, item):
    categroie_name =  [dt.get('name') for dt in lst if item == dt.get('id')][0]
    if len(categroie_name.split()) == 2:
        temp = categroie_name.split()
        categroie_name = temp[0] + '_' + temp[1]
    return categroie_name


gt_file_path = './ground-truth-split'
if not os.path.exists(gt_file_path):
    os.makedirs(gt_file_path)


for index, image_id in tqdm(enumerate(image_ids)):
    indexs = get_all_index(annotation_ids, image_id)
    with open('{}/{}.txt'.format(gt_file_path, file_names[index].split('.')[0]), 'w') as f:
        for idx in indexs:
            f.write(get_categroie_name(categroies, category_ids[idx]))
            f.write(' ')
            # change label
            bboxs[idx][2] = bboxs[idx][0] + bboxs[idx][2]
            bboxs[idx][3] = bboxs[idx][1] + bboxs[idx][3]
            f.write(' '.join(map(str, bboxs[idx])))
            f.write('\n')
