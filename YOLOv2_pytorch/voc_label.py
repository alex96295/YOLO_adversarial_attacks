import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# line 51 important for label file creation (format: 2007_test.txt)

sets=[('2012', 'train'), ('2012', 'val'), ('2008', 'train'), ('2008', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))

    # take the .xml file from the annotation folder and convert it to .txt file in the label folder only for the used images
    out_file = open('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')  # write a file with format: e.g. "2008_test.txt"
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))  # take the list_file (.txt) created and write on it the list of images.
        # the .txt list_files created are concatenated in the train.txt file
        convert_annotation(year, image_id)
    list_file.close()

