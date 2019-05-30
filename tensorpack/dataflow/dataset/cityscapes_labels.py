#!/usr/bin/python
#
# Cityscapes labels
#
# Courtesy of:
#  https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

from collections import namedtuple
import numpy as np

cityscapes_label_properties = ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances',
                               'ignore_in_eval', 'color']

CityscapesLabel = namedtuple('CityscapesLabel', cityscapes_label_properties)

cityscapes_labels = [
    #                  name        id tr_id   cat       catId hasInsts ignoreInEval color
    CityscapesLabel('unlabeled',    0,  255, 'void',        0, False, True,  (0, 0, 0)),
    CityscapesLabel('ego vehicle',  1,  255, 'void',        0, False, True,  (0, 0, 0)),
    CityscapesLabel('rectification '
                    'border',       2,  255, 'void',        0, False, True,  (0, 0, 0)),
    CityscapesLabel('out of roi',   3,  255, 'void',        0, False, True,  (0, 0, 0)),
    CityscapesLabel('static',       4,  255, 'void',        0, False, True,  (0, 0, 0)),
    CityscapesLabel('dynamic',      5,  255, 'void',        0, False, True,  (111, 74, 0)),
    CityscapesLabel('ground',       6,  255, 'void',        0, False, True,  (81, 0, 81)),
    CityscapesLabel('road',         7,  0,   'flat',        1, False, False, (128, 64, 128)),
    CityscapesLabel('sidewalk',     8,  1,   'flat',        1, False, False, (244, 35, 232)),
    CityscapesLabel('parking',      9,  255, 'flat',        1, False, True,  (250, 170, 160)),
    CityscapesLabel('rail track',   10, 255, 'flat',        1, False, True, (230, 150, 140)),
    CityscapesLabel('building',     11, 2,   'construction',2, False, False, (70, 70, 70)),
    CityscapesLabel('wall',         12, 3,   'construction',2, False, False, (102, 102, 156)),
    CityscapesLabel('fence',        13, 4,   'construction',2, False, False, (190, 153, 153)),
    CityscapesLabel('guard rail',   14, 255, 'construction',2, False, True,  (180, 165, 180)),
    CityscapesLabel('bridge',       15, 255, 'construction',2, False, True,  (150, 100, 100)),
    CityscapesLabel('tunnel',       16, 255, 'construction',2, False, True,  (150, 120, 90)),
    CityscapesLabel('pole',         17, 5,   'object',      3, False, False, (153, 153, 153)),
    CityscapesLabel('polegroup',    18, 255, 'object',      3, False, True,  (153, 153, 153)),
    CityscapesLabel('traffic light',19, 6,   'object',      3, False, False, (250, 170, 30)),
    CityscapesLabel('traffic sign', 20, 7,   'object',      3, False, False, (220, 220, 0)),
    CityscapesLabel('vegetation',   21, 8,   'nature',      4, False, False, (107, 142, 35)),
    CityscapesLabel('terrain',      22, 9,   'nature',      4, False, False, (152, 251, 152)),
    CityscapesLabel('sky',          23, 10,  'sky',         5, False, False, (70, 130, 180)),
    CityscapesLabel('person',       24, 11,  'human',       6, True,  False, (220, 20, 60)),
    CityscapesLabel('rider',        25, 12,  'human',       6, True,  False, (255, 0, 0)),
    CityscapesLabel('car',          26, 13,  'vehicle',     7, True,  False, (0, 0, 142)),
    CityscapesLabel('truck',        27, 14,  'vehicle',     7, True,  False, (0, 0, 70)),
    CityscapesLabel('bus',          28, 15,  'vehicle',     7, True,  False, (0, 60, 100)),
    CityscapesLabel('caravan',      29, 255, 'vehicle',     7, True,  True,  (0, 0, 90)),
    CityscapesLabel('trailer',      30, 255, 'vehicle',     7, True,  True,  (0, 0, 110)),
    CityscapesLabel('train',        31, 16,  'vehicle',     7, True,  False, (0, 80, 100)),
    CityscapesLabel('motorcycle',   32, 17,  'vehicle',     7, True,  False, (0, 0, 230)),
    CityscapesLabel('bicycle',      33, 18,  'vehicle',     7, True,  False, (119, 11, 32)),
    CityscapesLabel('license plate',-1, 13,  'vehicle',     7, False, True,  (0, 0, 142)),
]  # we handle the license plate differently (map it onto car), which was suggested by the creators.

# == Equivalent matrix for quick operations
cityscapes_label_dictionary = {key: np.array([label[key_idx] for label in cityscapes_labels])
                               for key_idx, key in enumerate(cityscapes_label_properties)}

cityscapes_classes_to_train = cityscapes_label_dictionary['ignore_in_eval'] == False


def map_cityscape_labels_to_train_labels(original_labels_img):
    """
    Warning: this is in-place!  Hope you didn't want the original labels...
    This is done in the fastest way I could think of, but it'll break if the
    original class id's change (changing the train_ids is fine).
    """
    # Weird trick since id is equivalent to index into the labels vectors (except license plate, -1)
    original_labels_img[original_labels_img == -1] = 34
    original_labels_img = cityscapes_label_dictionary['train_id'][original_labels_img]
    # for i, cls_idx in enumerate(label_dictionary['id']):
    #     label_idxs[cls_idx] = i
    # new_label = label_dictionary['train_id'][label_dictionary['id'] == label_idx]
    # assert len(new_label) == 1
    return original_labels_img


"""  Notes from the authors of the original code:
# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# Label properties:
'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
# We use them to uniquely name a class

'id'          , # An integer ID that is associated with this label.
# The IDs are used to represent the label in ground truth images
# An ID of -1 means that this label does not have an ID and thus
# is ignored when creating ground truth images (e.g. license plate).
# Do not modify these IDs, since exactly these IDs are expected by the
# evaluation server.

'train_id'     , # Feel free to modify these IDs as suitable for your method. Then create
# ground truth images with train IDs, using the tools provided in the
# 'preparation' folder. However, make sure to validate or submit results
# to our evaluation server using the regular IDs above!
# For trainIds, multiple labels might have the same ID. Then, these labels
# are mapped to the same class in the ground truth images. For the inverse
# mapping, we use the label that is defined first in the list below.
# For example, mapping all void-type classes to the same ID in training,
# might make sense for some approaches.
# Max value is 255!

'category'    , # The name of the category that this label belongs to

'category_id'  , # The ID of this category. Used to create ground truth images
# on category level.

'has_instances', # Whether this label distinguishes between single instances or not

'ignore_in_eval', # Whether pixels having this class as ground truth label are ignored
# during evaluations or not

'color'       , # The color of this label

"""
