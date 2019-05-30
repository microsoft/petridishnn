from ..base import RNGDataFlow
from ...utils import logger,fs
import os
import numpy as np

def load_data_from_npzs(fnames):
    if not isinstance(fnames, list):
        fnames = [fnames]
    Xs = []
    Ys = []
    for fname in fnames:
        d = np.load(fname)
        logger.info('Loading from {}'.format(fname))
        X, Y = (d['X'], d['Y'])
        Xs.append(X)
        Ys.append(Y)
    return np.stack(X), np.stack(Y)

class Camvid(RNGDataFlow):
    name = 'camvid'
    non_void_nclasses = 11
    _void_labels = [11]

    # optional arguments
    data_shape = (360, 480, 3)
    mean = [0.39068785, 0.40521392, 0.41434407]
    std = [0.29652068, 0.30514979, 0.30080369]

    _cmap = {
       0: (128, 128, 128),    # sky
       1: (128, 0, 0),        # building
       2: (192, 192, 128),    # column_pole
       3: (128, 64, 128),     # road
       4: (0, 0, 192),        # sidewalk
       5: (128, 128, 0),      # Tree
       6: (192, 128, 128),    # SignSymbol
       7: (64, 64, 128),      # Fence
       8: (64, 0, 128),       # Car
       9: (64, 64, 0),        # Pedestrian
       10: (0, 128, 192),     # Bicyclist
       11: (0, 0, 0)}         # Void

    _mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
           4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
           9: 'pedestrian', 10: 'byciclist', 11: 'void'}
    
    # frequency and weight of each class (including void)
    class_freq = np.array([ 0.16845114,  0.23258652,  0.00982927,  0.31658215,  0.0448627,
        0.09724055, 0.01172954, 0.01126809, 0.05865686, 0.00639231, 0.00291665, 0.03948423])
    class_weight = sorted(class_freq)[len(class_freq)//2] / class_freq
    #class_weight = np.array([  0.49470329,   0.35828961,   8.47807568,   0.26322815,
    #    1.8575192 ,   0.85698135,   7.10457224,   7.39551774,
    #    1.42069214,  13.03649617,  28.57158304,   2.11054735])
    
    def __init__(self, which_set, shuffle=True, pixel_z_normalize=True, data_dir=None,
        is_label_one_hot=False, 
        slide_all=False, slide_window_size=224, void_overlap=False):
        """
        which_set : one of train, val, test, trainval
        shuffle:
        data_dir: <data_dir> should contain train.npz, val.npz, test.npz
        """
        self.shuffle = shuffle
        self.pixel_z_normalize = pixel_z_normalize
        self.is_label_one_hot = is_label_one_hot
        self.void_overlap = void_overlap

        if data_dir is None:
            data_dir = fs.get_dataset_path('camvid')
        assert os.path.exists(data_dir)
        for set_name in ['train', 'val', 'test']:
            assert os.path.exists(os.path.join(data_dir, '{}.npz'.format(set_name)))

        assert which_set in ['train', 'val', 'test', 'trainval'],which_set
        if which_set == 'train':
            load_fns = ['train']
        elif which_set == 'val':
            load_fns = ['val']
        elif which_set == 'test':
            load_fns = ['test']
        else: #if which_set == 'trainval':
            load_fns = ['train', 'val'] 
        # These npz are assumed to have NHWC format for image, and NHW for label
        load_fns = map(lambda fn : os.path.join(data_dir, '{}.npz'.format(fn)), load_fns)

        self.X, self.Y = load_data_from_npzs(load_fns)
        assert self.X.dtype == 'uint8'

        self.slide_window_size = slide_window_size
        self.slide_all = slide_all
        self.slide_all_size =None 

    def get_data(self):
        idxs = np.arange(len(self.X))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            X = np.asarray(self.X[k], dtype=np.float32) / 255.0
            Y = self.Y[k]
            H,W = (X.shape[0], X.shape[1])
            void = Camvid._void_labels[0]
            if self.is_label_one_hot:
                K = Camvid.non_void_nclasses
                Y_tmp = np.zeros((H,W,K),dtype=np.float32) 
                mask = (Y.reshape([-1]) < K)
                Y_tmp.reshape([-1,K])[np.arange(H*W)[mask], Y.reshape([-1])[mask]] = 1.0
                Y = Y_tmp
                void = np.zeros(K)
            if self.pixel_z_normalize:
                X = (X - Camvid.mean) / Camvid.std
            if not self.slide_all:
                # do not slide all windows
                yield [X, Y]
            else:
                # slide all windows
                side = self.slide_window_size
                n_h = H // side + int(H % side != 0) 
                n_w = W // side + int(W % side != 0)
                for hi in range(n_h):
                    h_overlap = 0
                    row = hi*side
                    row_end = row+side
                    if row_end > H:
                        if self.void_overlap:
                            h_overlap = row - (H-side)
                        row = H - side 
                        row_end = H
                    for wi in range(n_w):
                        w_overlap = 0
                        col = wi*side
                        col_end = col+side
                        if col_end > W:
                            if self.void_overlap:
                                w_overlap = col - (W-side) 
                            col = W - side
                            col_end = W
                                
                        Xrc = X[row:row_end, col:col_end]
                        Yrc = Y[row:row_end, col:col_end].copy()
                        if h_overlap > 0:
                            Yrc[:h_overlap, :] = void
                        if w_overlap > 0:
                            Yrc[:, :w_overlap] = void

                        yield [Xrc, Yrc]
                

    def size(self):
        if not self.slide_all:
            return len(self.X)
        if self.slide_all_size is None:
            H, W = self.X.shape[1], self.X.shape[2]
            side = self.slide_window_size
            n_h = H // side + int(H % side !=0)
            n_w = W // side + int(W % side !=0)
            self.slide_all_size = n_h * n_w * len(self.X)
        return self.slide_all_size

    def stitch_sliding_images(self, l_imgs):
        """
            The l_imgs should be probability distribution of labels. 
        """
        side = self.slide_window_size
        H,W = (Camvid.data_shape[0], Camvid.data_shape[1])
        n_h = H // side + int(H % side != 0) 
        n_w = W // side + int(W % side != 0)
        assert n_h * n_w == len(l_imgs), len(l_imgs)

        n_ch = len(l_imgs[0].reshape([-1])) / side **2
        assert n_ch > 1, n_ch
        image = np.zeros((H, W, n_ch))
        i = -1
        for hi in range(n_h):
            row = hi * side
            row_end = row+side
            if row_end > H:
                row_end = H
                row = H - side
            for wi in range(n_w):
                col = wi*side
                col_end = col+side
                if col_end > W:
                    col_end = W
                    col = W - side
                i+=1
                r_ = row_end - row
                c_ = col_end - col
                window = l_imgs[i].reshape([side, side, n_ch])
                image[row:row_end, col:col_end] += window
        return image
                    


