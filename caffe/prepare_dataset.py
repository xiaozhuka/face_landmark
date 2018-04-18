# -*- coding: utf-8 -*-

"""Some operations.
"""
import numpy as np
import h5py

from cv2 import imread, resize

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from dataset import random_augmentation
from dlib_face_detection import dlib_file_list

def run(p=[r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\ce',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\hu',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\low'],
        pkl_f=os.path.join(r'E:\python_vanilla', 'result.pkl')):
    func = random_augmentation(dirty_circle=0.5,
                               random_flip_horizontal=0.5,
                               random_filp_vertical=0.5,
                               random_landmark_mask=None)

    img_lists = []
    labels_lists = []
    ds = dlib_file_list(p, pkl_f=pkl_f)
    np.random.shuffle(ds)
    filenames = [ds[i][0] for i in range(len(ds))]
    labels = [ds[i][1] for i in range(len(ds))]
    for filename, label in zip(filenames, labels):
        tmp = filename.split('\\')
        filename = '/home/cuishi/zcq/data/' + tmp[-2] + '/' + tmp[-1]

        if not os.path.isfile(filename):
            print('filename: %s don\' exists' % filename)
            continue
        img = imread(filename)
        if img is None:
            print('Failed to read file: %s'%filename)
            continue
        print('Read in image: %s'%filename)
        img = resize(img, (40, 40))
        label = np.array(label, dtype=np.float32)
        img = img.astype(np.float32)
        img = img / 255.0

        s = img.shape
        label[::2] = label[::2] / s[1]
        label[1::2] = label[1::2] / s[0]
        img_lists.append((img[:, :, ::-1] - np.mean(img)) / np.std(img))
        labels_lists.append(label)

        for _ in range(3):
            label = np.array(label, dtype=np.float32)
            img_ = img.copy()
            label_ = label.copy()
            img__, label__ = func([img_, label_])

            s = img__.shape
            label__[::2] = label__[::2] / s[1]
            label__[1::2] = label__[1::2] / s[0]

            img_lists.append((img__[:, :, ::-1] - np.mean(img__)) / np.std(img__))
            labels_lists.append(label__)

    bound = int(len(img_lists) * 0.8)
    train_imgs = img_lists[:bound]
    train_labels = labels_lists[:bound]

    test_imgs = img_lists[bound:]
    test_labels = labels_lists[bound:]

    with h5py.File('./train_dataset.h5', 'w') as f:
        f.create_dataset('data', data=train_imgs, dtype=np.float32)
        f.create_dataset('landmarks', data=train_labels, dtype=np.float32)

    with open('./train_dataset.txt', 'w') as f:
        f.write(os.path.abspath('./train_dataset.h5') + '\n')

    with h5py.File('./test_dataset.h5', 'w') as f:
        f.create_dataset('data', data=test_imgs, dtype=np.float32)
        f.create_dataset('landmarks', data=test_labels, dtype=np.float32)

    with open('./test_dataset.txt', 'w') as f:
        f.write(os.path.abspath('./test_dataset.h5') + '\n')

if __name__ == '__main__':
    run([r'/home/cuishi/zcq/data/25points_selected',
            r'/home/cuishi/zcq/data/ce',
            r'/home/cuishi/zcq/data/hu',
            r'/home/cuishi/zcq/data/low'],
        '/home/cuishi/zcq/python_vanilla/result.pkl2')






