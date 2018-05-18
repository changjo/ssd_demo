import cv2
import numpy as np


def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])


VOC_CLASSES = (
      'aeroplane', 'bicycle', 'bird', 'boat',
      'bottle', 'bus', 'car', 'cat', 'chair',
      'cow', 'diningtable', 'dog', 'horse',
      'motorbike', 'person', 'pottedplant',
      'sheep', 'sofa', 'train', 'tvmonitor')


COLORS = [rgb2bgr((0,     0,   0)),
        rgb2bgr((111,  74,   0)),
        rgb2bgr(( 81,   0,  81)),
        rgb2bgr((128,  64, 128)),
        rgb2bgr((244,  35, 232)),
        rgb2bgr((230, 150, 140)),
        #rgb2bgr((220, 220,   0)),
        rgb2bgr(( 70,  70,  70)),
        rgb2bgr((102, 102, 156)),
        rgb2bgr((190, 153, 153)),
        rgb2bgr((150, 120,  90)),
        rgb2bgr((153, 153, 153)),
        rgb2bgr((250, 170,  30)),
        rgb2bgr((220, 220,   0)),
        rgb2bgr((107, 142,  35)),
        rgb2bgr(( 52, 151,  52)),
        rgb2bgr(( 70, 130, 180)),
        rgb2bgr((220,  20,  60)),
        rgb2bgr((  0,   0, 142)),
        rgb2bgr((  0,   0, 230)),
        rgb2bgr((119,  11,  32))]


INDEX_TO_CLASS = dict(zip(range(len(VOC_CLASSES)), VOC_CLASSES)) 


def prop2abs(box, width, height):
    """
    Convert proportional center-width bounds to absolute min-max bounds
    """
    xmin, ymin, xmax, ymax, _ = box 
    xmin *= width
    ymin *= height
    xmax *= width
    ymax *= height
    return int(xmin), int(xmax), int(ymin), int(ymax)


def draw_box(img, box, label, color, conf=None):
    width, height = img.shape[1], img.shape[0]
    xmin, xmax, ymin, ymax = prop2abs(box, width, height)
    img_box = np.copy(img)
    #img_box = img
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if conf is None:
        cv2.putText(img_box, label, (xmin+5, ymin-5), font, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img_box, "{} {:.3f}".format(label, conf), (xmin+5, ymin-5), font, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
    #cv2.addWeighted(img_box, alpha, img, 1, 0, img)


def draw_boxes(img, box, label, color, conf=None):
    width, height = img.shape[1], img.shape[0]
    xmin, xmax, ymin, ymax = prop2abs(box, width, height)
    img_box = np.copy(img)
    #img_box = img
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if conf is None:
        cv2.putText(img_box, label, (xmin+5, ymin-5), font, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img_box, "{} {:.3f}".format(label, conf), (xmin+5, ymin-5), font, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
    #cv2.addWeighted(img_box, alpha, img, 1, 0, img)


def draw_images(image, boxes, confs=[], color_bgr2rgb=True):
    if 0:
        img = cv2.resize(image, (512, 512))
    if 1:
        img = image
    for i in range(len(boxes)):
        box = boxes[i]
        label_index = int(box[-1])
        conf = None
        if len(confs) > 0 :
            conf = confs[i]
        draw_box(img, box, INDEX_TO_CLASS[label_index], COLORS[label_index], conf=conf)
        #draw_box(img, box)
    img[img > 255] = 255
    img[img < 0] = 0
    if color_bgr2rgb:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        img = img.astype(np.uint8)

    return img

