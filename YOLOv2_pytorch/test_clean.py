import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import json

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  --use gpu only
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

cfgfile = "./cfg/yolo.cfg"
weightfile = "./weights/yolov2.weights"

imgdir = "../yolov4_pytorch/single_box_trial_yolov4/inria_few_data_img/"
savedir = "./single_box_trial_yolov2/yolov2_clean_fewdata_labels/"

def detect(cfgfile, weightfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = './data/voc.names'
    elif m.num_classes == 80:
        namesfile = './data/coco.names'
    else:
        namesfile = './data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    n = 0
    clean_results = []

    for imgfile in os.listdir(imgdir):
        print('\nIMAGE #' + str(n) + ': ' + os.path.splitext(imgfile)[0])
        print(n + 1)
        n += 1

        img_path = os.path.join(imgdir, imgfile)
        txt_clean_name = imgfile.replace('.png', '.txt')
        txt_clean_path = os.path.join(savedir, txt_clean_name)

        img = Image.open(img_path).convert('RGB')
        sized = img.resize((m.width, m.height))

        start = time.time()

        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        print(boxes)

        textfile = open(txt_clean_path, 'w+')
        for box in boxes:
            cls_id = box[5]
            if cls_id == 0:
                textfile.write(f'{cls_id} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')

                clean_results.append({'image_id': os.path.splitext(imgfile)[0], 'bbox': [(box[0].item() - box[2].item() / 2),
                                                                                         (box[1].item() - box[3].item() / 2),
                                                                                         box[2].item(),
                                                                                         box[3].item()],
                                      'score': box[4].item(),
                                      'category_id': 1})

        finish = time.time()
        print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        # class_names = load_class_names(namesfile)
        # plot_boxes(img, boxes, os.path.join(savedir, 'clean/', 'after_detection', imgfile), class_names)


    # with open('./json_files/clean_results.json', 'w') as fp:
    #     json.dump(clean_results, fp)


detect(cfgfile, weightfile)

