import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  --use gpu only
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

cfgfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/cfg/yolo.cfg'
weightfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/yolov2.weights'
imgfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/person_4.jpg'

def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/coco.names'
    else:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/names'
    
    use_cuda = 0
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/coco.names'
    else:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/names'
    
    use_cuda = 0
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/coco.names'
    else:
        namesfile = 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_VOC/data/names'
    
    use_cuda = 0
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

#if __name__ == '__main__':
#    if len(sys.argv) == 4:
#        cfgfile = sys.argv[1]
#        weightfile = sys.argv[2]
#        imgfile = sys.argv[3]
#        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
#    else:
#        print('Usage: ')
#        print('  python detect.py cfgfile weightfile imgfile')
#        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)

detect(cfgfile, weightfile, imgfile)
#detect_cv2(cfgfile, weightfile, imgfile)
#detect_skimage(cfgfile, weightfile, imgfile)
