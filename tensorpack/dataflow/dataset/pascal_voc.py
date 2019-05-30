from ..base import RNGDataFlow
import numpy as np

class PascalVOC(RNGDataFlow):
    name = 'pascal_voc'
    non_void_nclasses = 21
    _void_labels = [255]

    mean = [122.67891434, 116.66876762, 104.00698793]
    std = [70.19812053, 69.35514486, 72.71005314]

    class_weight = np.asarray([\
        0.01564238,  1.27773941,  1.42356741,  1.14417922,  1.60704982,
        2.1443634 ,  0.80881667,  0.55887061,  0.34558904,  0.85274589,
        1.7184062 ,  1.03832972,  0.41796997,  1.15941226,  0.90510136,
        0.16320343,  1.71444452,  1.56673861,  0.86603409,  0.77613521,
        1.45208943,  0.96439946], dtype=np.float32)

    GTclasses = list(range(21)) + [255]
    _cmap = {
        0: (0, 0, 0),           # background
        1: (255, 0, 0),         # aeroplane
        2: (192, 192, 128),     # bicycle
        3: (128, 64, 128),      # bird
        4: (0, 0, 255),         # boat
        5: (0, 255, 0),         # bottle
        6: (192, 128, 128),     # bus
        7: (64, 64, 128),       # car
        8: (64, 0, 128),        # cat
        9: (64, 64, 0),         # chair
        10: (0, 128, 192),      # cow
        11: (0, 255, 255),      # diningtable
        12: (255, 0, 255),      # dog
        13: (255, 128, 0),      # horse
        14: (0, 102, 102),      # motorbike
        15: (102, 0, 204),      # person
        16: (128, 255, 0),      # potted_plant
        17: (224, 224, 224),    # sheep
        18: (102, 0, 51),       # sofa
        19: (153, 76, 0),       # train
        20: (229, 244, 204),    # tv_monitor
        255: (255, 255, 255)    # void
    }
    _mask_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
                    4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
                    9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
                    13: 'horse', 14: 'motorbike', 15: 'person',
                    16: 'potted_plant', 17: 'sheep', 18: 'sofa',
                    19: 'train', 20: 'tv_monitor', 255: 'void'}
