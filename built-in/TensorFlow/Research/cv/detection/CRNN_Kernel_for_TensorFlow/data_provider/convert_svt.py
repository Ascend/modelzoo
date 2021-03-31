from PIL import Image 
import numpy as np 
from xml.etree import ElementTree as ET
import argparse 
import os 


def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-d', '--dataset_dir', type=str,default='./',
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-x', '--xml_file', type=str,default='test.xml',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--output_dir', type=str,default='./processed',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-a', '--output_annotation', type=str,default='./annotation.txt',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-l', '--output_lexicon', type=str,default='./lexicon.txt',
                        help='Directory where ord map dictionaries for the dataset were stored')
    return parser.parse_args()



def xml_to_dict(xml_file, save_file=False):
    
    tree = ET.parse(xml_file)
    root = tree.getroot() 
    
    children = root.getchildren()
    imgs_labels = []

    for ch in children:
        im_label = {}
        
        for ch01 in ch.getchildren():
            if ch01.tag in "address":
                continue
            elif ch01.tag in 'taggedRectangles':
                # multiple children
                rect_list = []
                #rect = {}
                for ch02 in ch01.getchildren():
                    rect = {}
                    rect['location'] = ch02.attrib
                    rect['label'] = ch02.getchildren()[0].text
                    #print(rect['label'])
                    rect_list.append(rect)
                #print("number of rect : ", len(rect_list))
                im_label['rect'] = rect_list
            else:
                im_label[ch01.tag] = ch01.text
        imgs_labels.append(im_label)

    if save_file:
        np.save("annotation_train.npy",imgs_labels)

    return imgs_labels    


def image_crop_save(image,location, output_dir):
    '''crop image with location (h,w,x,y)
       save cropped image to output directory 
       
    '''
    start_x = location[2]
    end_x = start_x + location[1]
    start_y = location[3]
    if start_y<0:
        start_y=0
    end_y = start_y + location[0]
    #print("image array shape :{}".format(image.shape))
    #print("crop region ", start_x, end_x,start_y,end_y)
    if len(image.shape)==3:
        cropped = image[start_y:end_y,start_x:end_x,:]
    else: 
        cropped = image[start_y:end_y,start_x:end_x]
    im = Image.fromarray(np.uint8(cropped))
    im.save(output_dir)


def convert():
    args = init_args()
    if not os.path.exists(args.dataset_dir):
        raise ValueError("dataset_dir :{ } does not exist".format(args.dataset_dir))

    if not os.path.exists(args.xml_file):
        raise ValueError("xml_file :{ } does not exist".format(args.xml_file))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ims_labels_dict = xml_to_dict(args.xml_file,True)
    num_images = len(ims_labels_dict)
    lexicon_list = []
    annotation_list = []
    print("Converting annotation, {} images in total ".format(num_images))
    for i in range(num_images):
        img_label = ims_labels_dict[i]
        image_name = img_label['imageName']
        lex = img_label['lex']
        rects = img_label['rect']
        name, ext = image_name.split('.')
        name = name.replace('/','_')
        fullpath = os.path.join(args.dataset_dir,image_name)
        im_array = np.asarray(Image.open(fullpath))
        lexicon_list.append(lex)
        print("processing image: {}".format(image_name))
        for j in range(len(rects)):
            rect = rects[j]
            location = rect['location']
            h = int(location['height'])
            w = int(location['width'])
            x = int(location['x'])
            y = int(location['y']) 
            label = rect['label']
            loc = [h,w,x,y]
            output_name = name+"_"+str(j)+"_"+label+'.'+ext
            output_file = os.path.join(args.output_dir,output_name)
            
            image_crop_save(im_array,loc,output_file) 
            ann = output_name+","+label+','+str(i)
            print(ann)
            annotation_list.append(ann)
     
    ann_file = args.output_annotation

    with open(ann_file,'w') as f:
        for line in annotation_list:
            txt = line+'\n'
            f.write(txt)



if __name__=="__main__":
    
    convert() 





       
    













#xml_to_dict('test.xml')

