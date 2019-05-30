import numpy as np
import os
import cv2
import multiprocessing

from tensorpack.dataflow import DataFlow, PrefetchDataZMQ, \
    LMDBData, LMDBDataPoint, PrefetchData, \
    MapDataComponent, AugmentImageComponent, BatchData
from tensorpack.dataflow import imgaug
from tensorpack.dataflow.serialize import LMDBSerializer
dump_dataflow_to_lmdb = LMDBSerializer.save


class TinyImageNetInfo(object):

    def __init__(self, data_dir):
        """
        Assume that in data_dir, wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
        and unzip tiny-imagenet-200.zip just happened.

        Now we process the individual images and combine them into LMDB

        Args:
        data_dir : location where the unzip xxx.zip happened. e.g.,
            ls $data_dir should have: "tiny-imagenet-200"
            and ls $data_dir/tiny-imagenet-200 should have
                test/  train/  val/  wnids.txt  words.txt
        """
        src_root = os.path.join(data_dir, 'tiny-imagenet-200')
        assert os.path.exists(src_root), src_root

        # meta info : code name and zero-based label index mapping
        wnids_fn = os.path.join(src_root, 'wnids.txt')
        with open(wnids_fn, 'rt') as fin:
            l_code_names = []
            code_name_to_index = dict()
            for li, line in enumerate(fin):
                line = line.strip()
                l_code_names.append(line)
                code_name_to_index[line] = li
        self.code_name_to_index = code_name_to_index
        self.l_code_names = l_code_names

        # mapping for split (train/val/test) to list of paths/label indices
        self.img_list = dict()
        self.label_list = dict()

        # meta info : the list of images for each split
        # train split:
        l_train_paths = []
        l_train_labels = []
        for label, code_name in enumerate(l_code_names):
            label_img_root = os.path.join(src_root, 'train', code_name, 'images')
            l_img_basenames = os.listdir(label_img_root)
            l_train_paths.extend(\
                [ os.path.join(label_img_root, basename) for basename in l_img_basenames ])
            l_train_labels.extend(\
                [ label for _ in range(len(l_img_basenames)) ])
        self.img_list['train'] = l_train_paths
        self.label_list['train'] = l_train_labels

        # val split
        l_val_paths = []
        l_val_labels = []
        with open(os.path.join(src_root, 'val', 'val_annotations.txt'), 'rt') as fin:
            for line in fin:
                line_info = line.strip().split()
                try:
                    img_basename, code_name = line_info[0], line_info[1]
                    img_fn = os.path.join(src_root, 'val', 'images', img_basename)
                    assert os.path.exists(img_fn), img_fn
                    label = code_name_to_index[code_name]
                    l_val_paths.append(img_fn)
                    l_val_labels.append(label)
                except Exception as e:
                    print("Error during preprocessing tiny imagenet validation set : {}".format(e))
                    return
        self.img_list['val'] = l_val_paths
        self.label_list['val'] = l_val_labels

        # test split
        test_img_root = os.path.join(src_root, 'test', 'images')
        l_basenames = os.listdir(test_img_root)
        l_test_paths = [ os.path.join(test_img_root, basename) for basename in l_basenames ]
        self.img_list['test'] = l_test_paths
        n_labels = len(l_code_names)
        self.label_list['test'] = [n_labels for _ in range(len(l_test_paths))]


class RawTinyImageNet(DataFlow):

    def __init__(self, split, shuffle=True, meta_info=None, data_dir=None):
        assert data_dir is not None or meta_info is not None, \
            'One of meta_info and data_dir needs to be specified'
        if meta_info is None:
            meta_info = TinyImageNetInfo(data_dir)

        self.img_list = meta_info.img_list[split]
        self.label_list = meta_info.label_list[split]
        self.shuffle = shuffle
        self.name = split

    def get_data(self):
        indices = list(range(self.size()))
        if self.shuffle:
            np.random.shuffle(indices)

        for k in indices:
            fn, label = self.img_list[k], self.label_list[k]
            with open(fn, 'rb') as fin:
                jpeg = fin.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]

    def size(self):
        return len(self.img_list)


def get_tiny_imagenet_augmented_data(subset, options,
        do_multiprocess=True, is_train=None, shuffle=None):
    isTrain = is_train if is_train is not None else (subset == 'train' and do_multiprocess)
    shuffle = shuffle if shuffle is not None else isTrain

    lmdb_path = os.path.join(options.data_dir,
        'tiny_imagenet_lmdb', 'tiny_imagenet_{}.lmdb'.format(subset))
    # since tiny imagenet is small (200MB zipped) we can shuffle all directly.
    # we skipped the LocallyShuffleData and PrefetchData routine.
    ds = LMDBData(lmdb_path, shuffle=shuffle)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    img_size = 64
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
                    targetArea = self.rng.uniform(0.3, 1.0) * area
                    aspectR = self.rng.uniform(0.75, 1.333)
                    ww = int(np.sqrt(targetArea * aspectR))
                    hh = int(np.sqrt(targetArea / aspectR))
                    if self.rng.uniform() < 0.5:
                        ww, hh = hh, ww
                    if hh <= h and ww <= w:
                        x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                        y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                        out = img[y1:y1 + hh, x1:x1 + ww]
                        out = cv2.resize(out, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                        return out
                out = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
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
            imgaug.ResizeShortestEdge(72),
            imgaug.CenterCrop((img_size, img_size)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, options.batch_size // options.nr_gpu, remainder=not isTrain)
    if do_multiprocess:
        ds = PrefetchData(ds, nr_prefetch=4, nr_proc=4)
    return ds


if __name__ == '__main__':
    """
    mkdir -p $GLOBAL_DATA_DIR/tiny_imagenet_raw
    cd $GLOBAL_DATA_DIR/tiny_imagenet_raw
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip

    cd $CODE_DIR/petridishnn
    mkdir -p $GLOBAL_DATA_DIR/tiny_imagenet_lmdb
    python2 petridish/data/tiny_imagenet.py
    """
    data_root = os.environ.get('GLOBAL_DATA_DIR', None)
    raw_data_root = os.environ.get("GLOBAL_DATA_RAW_DIR", None)
    if data_root is None:
        raise Exception("Data dir is not set. Set environ GLOBAL_DATA_DIR")
    data_dir = os.path.join(raw_data_root, 'tiny_imagenet_raw')
    lmdb_data_dir = os.path.join(data_root, 'tiny_imagenet_lmdb')

    assert os.path.exists(data_dir), data_dir
    assert os.path.exists(lmdb_data_dir), lmdb_data_dir

    meta_info = TinyImageNetInfo(data_dir)
    np.random.seed(19921102)

    for split in ['train', 'val', 'test']:
        shuffle = (split != 'test')
        ds_raw = RawTinyImageNet(split=split, shuffle=shuffle, meta_info=meta_info)
        ds_prefetch = PrefetchDataZMQ(ds_raw, nr_proc=1)
        dump_dataflow_to_lmdb(ds_prefetch,
            os.path.join(lmdb_data_dir, 'tiny_imagenet_{}.lmdb'.format(split)))
