# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from darknet import Darknet
from utils import *
import torch
import json
from torchvision import transforms
from load_data import *
import matplotlib.pyplot as plt

def generate_patch(type, patch_size):
    if type == 'gray':
        adv_patch_cpu = torch.full((3, patch_size, patch_size), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, patch_size, patch_size))

    return adv_patch_cpu


def pad_and_scale(img, lab, common_size):  # this method for taking a non-square image and make it square by filling the difference in w and h with gray

    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(img, (int(padding), 0))
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h / w)
    resize = transforms.Resize((common_size, common_size))  # make a square image of dim 416 x 416
    padded_img = resize(padded_img)  # choose here
    return padded_img, lab

def remove_pad(w_orig, h_orig, in_img):

        w = w_orig
        h = h_orig

        img = transforms.ToPILImage('RGB')(in_img)

        dim_to_pad = 1 if w < h else 2

        if dim_to_pad == 1:
            padding = (h - w) / 2
            #padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            #padded_img.paste(img, (int(padding), 0))
            image = Image.Image.resize(img, (h, h))
            image = Image.Image.crop(image, (int(padding), 0, int(padding) + w, h))

        else:
            padding = (w - h) / 2
            # padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            # padded_img.paste(img, (0, int(padding)))
            image = Image.Image.resize(img, (w, w))
            image = Image.Image.crop(image, (0, int(padding), w, int(padding) + h))

        return image



use_cuda = 1

cfgfile = "./cfg/yolo.cfg"
weightfile = "./weights/yolov2.weights"
imgdir = "../yolov4_pytorch/single_box_trial_yolov4/inria_few_data_img/"
#savedir = "./test_results_mytrial/"

#patchfile = "./patches/ale_obj_noepochloss_500ep.jpg" 

square_size = 416 # can be whatever, because I remove the pad
patch_size = 300

#patch_img = Image.open(patchfile).convert('RGB')
# plt.imshow(patch_img)
# plt.show()
#patch_img = patch_img.resize((patch_size,patch_size))
#adv_patch = transforms.ToTensor()(patch_img) # already in range 0,1
# print(adv_patch.type())

adv_patch = generate_patch('gray', patch_size)

m = Darknet(cfgfile)
m.load_weights(weightfile)
print('Loading weights from %s... Done!' % (weightfile))

num_classes = 80
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/names'


if use_cuda:
    m.cuda()

n=0
clean_results = []

for imgfile in os.listdir(imgdir):

    print('\nIMAGE #' + str(n) + ': ' + os.path.splitext(imgfile)[0])
    print(n+1)
    n+=1
    
    name = os.path.splitext(imgfile)[0]
    img_path = os.path.join(imgdir, imgfile)
    txt_clean_name = imgfile.replace('.png','.txt')
    patch_label_name = os.path.splitext(imgfile)[0] + '_p.txt'

    #patch_label_path = os.path.join(savedir, 'ensemble_yv3_ssdmbntv1_ssdlitembntv2_patches/proper_patched_obj_cls/', 'yolov4-labels/', patch_label_name)
    txt_clean_path = os.path.join('./single_box_trial_yolov2_random/yolov2_clean_fewdata_labels/', txt_clean_name)
    
    # create folders if they do not exists
    single_img_labdir = os.path.join('./single_box_trial_yolov2_random/yolov2_patched_fewdata_labels/', name, 'patch/')
    if os.path.isdir(single_img_labdir) == False:
        os.makedirs(single_img_labdir)

    img = Image.open(img_path).convert('RGB')

    h_orig = img.size[1]  # for opencv it is (height, width) and .shape, while for PIL it is (width, height) and .size
    w_orig = img.size[0]

    print('Start reading generated label file used for patch application')
    # read this label file back as a tensor
    textfile = open(txt_clean_path, 'r')
    if os.path.getsize(txt_clean_path):  # check to see if label file contains data.
        label = np.loadtxt(textfile)
        label_nopad = label
        # print(label.shape)
    else:
        label = np.ones([6])
        label_nopad = label

    if np.ndim(label) == 1:
        # label = label.unsqueeze(0)
        label = np.expand_dims(label, 0)
        label_nopad = label

    label = torch.from_numpy(label).float()
    label_nopad = torch.from_numpy(label_nopad).float()
    
    print('label file used for patch application read correctly')

    print('Start image preprocessing')
    image_clean_ref = img

    image_p, label_p = pad_and_scale(image_clean_ref, label, common_size=square_size)

    # convert image back to torch tensor
    image_tens = transforms.ToTensor()(image_p)

    # add fake batch size, fake because it has size = 1, so it's a single image (i.e you don't really need)
    img_fake_batch = torch.unsqueeze(image_tens, 0)
    lab_fake_batch = torch.unsqueeze(label_p, 0)
    lab_fake_batch_nopad = label_nopad.unsqueeze(0)
    
    # start splitting label clean file for single patch application
    lab_fake_batch_boxsplit = torch.unbind(lab_fake_batch, axis=1)  # to get each box line separated for patch application
    lab_fake_batch_boxsplit_nopad = lab_fake_batch_nopad.unbind(1)

    single_img_labdir_c = os.path.join('./single_box_trial_yolov2_random/yolov2_patched_fewdata_labels/', name, 'clean/')
    if os.path.isdir(single_img_labdir_c) == False:
        os.makedirs(single_img_labdir_c)

    dict_imageperbox_list = []
    #im_outperbox_list = []
    k = 1

    for single_box_label, single_box_label_nopad in zip(lab_fake_batch_boxsplit, lab_fake_batch_boxsplit_nopad):
        single_box_clean_name = name + '_box' + str(k) + '-c.txt' # nome con cui salverò l'immagine pulita
        single_box_clean_path = os.path.join(single_img_labdir_c, single_box_clean_name)

        textfile_clean = open(single_box_clean_path, 'w+')
        textfile_clean.write(
            f'{single_box_label_nopad[0][0]} {single_box_label_nopad[0][1]} {single_box_label_nopad[0][2]} {single_box_label_nopad[0][3]} {single_box_label_nopad[0][4]} {single_box_label_nopad[0][5]}')

        single_box_label = torch.unsqueeze(single_box_label, axis=1)

        box_name = name + '_box' + str(k) + '.jpg'
        k += 1

        adv_batch_t = PatchTransformer()(adv_patch, single_box_label, square_size, do_rotate=True, rand_loc=False)
    
        # adv_batch_im = transforms.ToPILImage('RGB')(adv_batch_t[0][0])
        # plt.imshow(adv_batch_im)
        # plt.show()
    
        p_img_batch = PatchApplier()(img_fake_batch, adv_batch_t)
    
        p_img = torch.squeeze(p_img_batch, 0)
    
        # come back to original dimensions
        p_img_orig = remove_pad(w_orig, h_orig, p_img)
        # plt.imshow(p_img_orig)
        # plt.show()
    
        out_path_part = "./single_box_trial_yolov2_random/patch_applied_images_no_det/"
        out_path = os.path.join(out_path_part, box_name)
        p_img_orig.save(out_path) # save patched image in folder
        # plt.imshow(p_img_orig)
        # plt.show()

        sized = p_img_orig.resize((m.width, m.height))
    
        start = time.time()
    
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        
        class_names = load_class_names(namesfile)
        out_path_part_detection = "./single_box_trial_yolov2_random/patch_applied_images_det/"
        out_path_detection = os.path.join(out_path_part_detection, box_name)
        plot_boxes(p_img_orig, boxes, out_path_detection, class_names)

        dict_imageperbox_list.append(boxes)

    for i, dictio in enumerate(dict_imageperbox_list):
        box_name = name + '_box' + str(i + 1) + '.txt'
    
        txtpath_p = os.path.join(single_img_labdir, box_name)
    
        textfile = open(txtpath_p, 'w+')
        for box in dictio:
            cls_id = box[5]
            if cls_id == 0:
                textfile.write(f'{cls_id} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')
    
                # clean_results.append({'image_id': name, 'bbox': [(box[0].item() - box[2].item() / 2),
                #                                                  (box[1].item() - box[3].item() / 2),
                #                                                  box[2].item(),
                #                                                  box[3].item()],
                #                       'score': box[4].item(),
                #                       'category_id': 1})
    
        finish = time.time()
        print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

   





