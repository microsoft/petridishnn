# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import os
import cv2
import json
import multiprocessing

from tensorpack.dataflow import DataFlow, PrefetchDataZMQ, \
    LMDBData, LMDBDataPoint, PrefetchData, \
    MapDataComponent, AugmentImageComponent, BatchData
from tensorpack.dataflow import imgaug, LocallyShuffleData
from tensorpack.utils import logger

from tensorpack.dataflow.serialize import LMDBSerializer
dump_dataflow_to_lmdb = LMDBSerializer.save

class INatRawData(DataFlow):

    def __init__(
            self,
            inat_dir, anno_json_fn,
            part=None, part_include=True,
            shuffle=True,
            allowed_labels=None):
        """
        Some notes about the Inat data (applies to 2017 and 2018 sets):
        1. Train and val have the same data directory structure and
        are co-located. Since there is no public test set, the val set here
        is considered the blind set and no parameters should be tuned on it
        2. Labels are in the original data path for both.
        3. Image list is already shuffled. Labels are 0-based.

        Args:
        inat_dir (str) : dirname for the root of inat raw data.
            It should have the following structure:
            <inat_dir>
                <train.json>
                <val.json>
                <train_val_imagedir>/
                    Actinopterygii/
                    ...
        anno_json_fn (str) : Filename for the train/val json file
        part (None or int) : Partition of the total dataset
        part_include (bool) : Whether the partition is to be included (True) or not (False)
        shuffle (bool) : whether to shuffle data every epoch.
        allowed_labels (dict of int to int) :
            map from the original label index to new label index.
            None means identity mapping for all labels.
        """
        assert os.path.exists(inat_dir), inat_dir

        json_fn = os.path.join(inat_dir, anno_json_fn)
        with open(json_fn) as fin:
            jd = json.load(fin)

        self.inat_dir = inat_dir
        self.shuffle = shuffle
        self.meta_info = []
        self.allowed_labels = allowed_labels

        tot_images = len(jd['images'])
        assert len(jd['annotations']) == tot_images

        cv_K = 10
        part_unit = int(tot_images/cv_K)

        #Here part is used for cros-validation
        #A positive number indicates extracting that part
        #A negative number indicates extracting the rest other than that part

        cv = 0
        if part is not None:
            cv = abs(part)
            assert cv <= cv_K
            #part_unit = 150000
            cv_start = cv * part_unit
            cv_end = min(tot_images, (cv+1) * part_unit)
            print(cv_start, cv_end)

        #need to shuffle before creating the parts
        #This needs to be a deterministic shuffle for the cv splits to be meaningful
        merged_data = list(zip(jd['images'], jd['annotations']))
        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(merged_data)

        # create meta data info
        for idx, (info, anno) in enumerate(merged_data):
            if part is not None:
                #Include only cv-split part (used for cv-val set)
                if part_include:
                    if idx < cv_start:
                        continue
                    if idx >= cv_end:
                        break
                #Exclude cv-split part (used for cv-train set)
                else:
                    if (idx >= cv_start) and (idx < cv_end):
                        continue

            fn = info['file_name']
            label = anno['category_id']
            if self.allowed_labels is None or label in self.allowed_labels:
                self.meta_info.append((fn, self.allowed_labels[label]))
        print(len(self.meta_info))

    def get_data(self):
        if self.shuffle:
            np.random.shuffle(self.meta_info)

        for info in self.meta_info:
            fn, label = info
            full_fn = os.path.join(self.inat_dir, fn)
            with open(full_fn, 'rb') as fin:
                jpeg = fin.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]

    def size(self):
        return len(self.meta_info)

image_mean = [0.485, 0.456, 0.406][::-1]
image_std = [0.229, 0.224, 0.225][::-1]

def get_inat_augmented_data(
        subset, options,
        lmdb_dir=None,
        year='2018',
        do_multiprocess=True,
        do_validation=False,
        is_train=None,
        shuffle=None,
        n_allow=None):
    input_size = options.input_size if options.input_size else 224
    isTrain = is_train if is_train is not None else (subset == 'train' and do_multiprocess)
    shuffle = shuffle if shuffle is not None else isTrain
    postfix = "" if n_allow is None else "_allow_{}".format(n_allow)

    #TODO: Parameterize the cv split to be consider
    #Currently hardcoding to 1
    cv = 1

    # When do_validation is True it will expect *cv_train and *cv_val lmdbs
    # Currently the cv_train split is always used
    if isTrain:
        postfix += '_cv_train_{}'.format(cv)
    elif do_validation:
        subset = 'train'
        postfix += '_cv_val_{}'.format(cv)

    if lmdb_dir == None:
        lmdb_path = os.path.join(
            options.data_dir, 'inat_lmdb', 'inat2018_{}{}.lmdb'.format(subset, postfix))
    else:
        lmdb_path = os.path.join(
            options.data_dir, lmdb_dir, 'inat{}_{}{}.lmdb'.format(year,subset, postfix))

    ds = LMDBData(lmdb_path, shuffle=False)
    if shuffle:
        ds = LocallyShuffleData(ds, 1024*80)  # This is 64G~80G in memory images
    ds = PrefetchData(ds, 1024*8, 1) # prefetch around 8 G
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0) # BGR uint8 data
    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            """
            crop 8%~100% of the original image
            See `Going Deeper with Convolutions` by Google.
            """
            def _augment(self, img, _):
                h, w = img.shape[:2]
                area = h * w
                for _ in range(10):
                    targetArea = self.rng.uniform(0.08, 1.0) * area
                    aspectR = self.rng.uniform(0.75, 1.333)
                    ww = int(np.sqrt(targetArea * aspectR))
                    hh = int(np.sqrt(targetArea / aspectR))
                    if self.rng.uniform() < 0.5:
                        ww, hh = hh, ww
                    if hh <= h and ww <= w:
                        x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                        y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                        out = img[y1:y1 + hh, x1:x1 + ww]
                        out = cv2.resize(
                            out, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                        return out
                out = cv2.resize(
                    img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                return out

        augmentors = [
            Resize(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256),
            imgaug.CenterCrop((input_size, input_size)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    if do_multiprocess:
        ds = PrefetchDataZMQ(ds, min(24, multiprocessing.cpu_count()))
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    return ds



def inat_raw_to_lmdb(
        inat_dir=None,
        inat_lmdb_dir=None,
        allowed_labels=None,
        year='2018',
        do_crossval=False,
        splits=['train', 'val']):
    if inat_dir is None or inat_lmdb_dir is None:
        try:
            data_root = os.environ['GLOBAL_DATA_DIR']
            if inat_dir is None:
                inat_dir = os.path.join(data_root, 'inat')
            if inat_lmdb_dir is None:
                inat_lmdb_dir = os.path.join(data_root, 'inat_lmdb')
        except:
            logger.info('GLOBAL_DATA_DIR is not set as an env variable')
            raise
    if not os.path.exists(inat_dir):
        raise Exception("inat dir does not exist")
    if not os.path.exists(inat_lmdb_dir):
        os.makedirs(inat_lmdb_dir)

    if allowed_labels is not None:
        label_map_fn = os.path.join(
            inat_lmdb_dir, 'label_map_{}.npz'.format(len(allowed_labels)))
        np.savez(label_map_fn, allowed_labels=allowed_labels)

    for split in splits:
        is_train = split == 'train'
        postfix = ''
        if allowed_labels:
            postfix += "_allow_{}".format(len(allowed_labels))
        anno_fn = '{}{}.json'.format(split, year)

        if is_train and do_crossval:
            cv_K = 10
            for cv in range(cv_K):
                postfix_cv_train = postfix + '_cv_train_{}'.format(cv)
                postfix_cv_val = postfix + '_cv_val_{}'.format(cv)
                #Train part of cv
                ds0 = INatRawData(
                    inat_dir,
                    anno_fn,
                    part=cv,
                    part_include=False,
                    allowed_labels=allowed_labels,
                    shuffle=True)
                ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
                dump_dataflow_to_lmdb(
                    ds1,
                    os.path.join(
                        inat_lmdb_dir,
                        'inat{}_{}{}.lmdb'.format(year, split, postfix_cv_train)
                    )
                )
                #Val part of cv
                ds2 = INatRawData(
                    inat_dir,
                    anno_fn,
                    part=cv,
                    part_include=True,
                    allowed_labels=allowed_labels,
                    shuffle=False)
                ds3 = PrefetchDataZMQ(ds2, nr_proc=1)
                dump_dataflow_to_lmdb(
                    ds3,
                    os.path.join(
                        inat_lmdb_dir,
                        'inat{}_{}{}.lmdb'.format(year, split, postfix_cv_val)
                    )
                )
        else:
            ds0 = INatRawData(
                    inat_dir,
                    anno_fn,
                    part=None,
                    allowed_labels=allowed_labels,
                    shuffle=is_train)
            ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
            dump_dataflow_to_lmdb(
                ds1,
                os.path.join(
                    inat_lmdb_dir,
                    'inat{}_{}{}.lmdb'.format(year, split, postfix)
                )
            )


if __name__ == '__main__':
    n_categories = 5089
    n_allowed = 1000 # 100

    #inat_raw_to_lmdb()

    np.random.seed(19921102)

    allowed_labels = list(np.random.choice(range(n_categories), n_allowed, replace=False))
    orig_to_new = dict()
    for idx, label in enumerate(allowed_labels):
        orig_to_new[label] = idx
    inat_raw_to_lmdb(inat_dir='/media/data/saurajim/inat2017_data', inat_lmdb_dir='/media/data/saurajim/inat2017_data/lmdb', allowed_labels=orig_to_new, year='2017', do_crossval=True)
