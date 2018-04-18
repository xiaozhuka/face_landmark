import os
import dlib
from skimage import io
import numpy as np
detector = dlib.get_frontal_face_detector()
import pickle


def _file_list(path=[r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\ce',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\hu',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\low']):

    ds = []
    for p in path:
        for f in os.listdir(p):
            if f.endswith('txt'):
                img_p = os.path.join(p, f[:-4]+'_0.jpg')
                if os.path.isfile(img_p):
                    with open(os.path.join(p, f), 'r') as f:
                        for _ in range(4):
                            f.readline()
                        tmp_lab = f.readline()
                        if tmp_lab.startswith('sensetime_21_points'):
                            tmp_lab = [float(x) for x in tmp_lab.split()[1:]]
                            ds.append([img_p, tmp_lab])
    return ds

def dlib_file_list(ipath=[r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\ce',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\hu',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\low'], pkl_f=None):
    file_l = []

    if pkl_f is None:
        pkl_f = os.path.join(r'E:\python_vanilla', 'result.pkl')
    if os.path.isfile(pkl_f):
        with open(pkl_f, 'rb') as f:
            file_l = pickle.load(f)

        return file_l

    for ds in _file_list(ipath):
        path = ds[0]
        landmark = ds[1]
        img = io.imread(path)

        dets = detector(img, 1)
        if len(dets) == 0:
            continue

        l, h, t, w = [dets[0].left(), dets[0].height(), dets[0].top(), dets[0].width()]

        landmark[::2] = [tmp - l for tmp in landmark[::2]]
        landmark[1::2] = [tmp - t for tmp in landmark[1::2]]

        if np.any(np.array(landmark) < 0):
            continue
        new_img = img[t:(t+w), l:(l+h)]
        new_img = new_img.astype(np.uint8)
        new_img_path = os.path.join(os.path.dirname(path), 'dlib_' + os.path.basename(path))

        try:
            io.imsave(new_img_path, new_img)
        except:
            pass

        file_l.append([new_img_path, landmark])
    with open(pkl_f, 'wb') as f:
        pickle.dump(file_l, f)
    return file_l


def detection_dlib(parent):
    for p in os.listdir(parent):
        if not p.endswith('.jpg'):
            continue

        img = io.imread(os.path.join(parent, p))

        dets = detector(img, 1)
        if len(dets) == 0:
            continue

        l, h, t, w = [dets[0].left(), dets[0].height(), dets[0].top(), dets[0].width()]

        new_img = img[t:(t+w), l:(l+h)]
        new_img = new_img.astype(np.uint8)
        new_img_path = os.path.join(parent, 'dlib_' + p)

        try:
            io.imsave(new_img_path, new_img)
        except:
            pass

if __name__ == '__main__':
    # dlib_file_list()
    detection_dlib(r'E:\fld_result\facepose')