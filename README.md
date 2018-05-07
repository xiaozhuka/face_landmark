# face landmark with pure cnn model by Tensorflow
*I moved my work to caffe using c++, see https://github.com/JunrQ/landmark_cpp*

This is my intern project in Shanghai Hongmu.

## Model
eight conv layers, see model.py for detail

## Dataset
random augmentation, see dataset.py for detail

## Result
I run 3000+ images forward through the network in less 1 seconds *GeForce GTX 1070*.

![result](https://github.com/JunrQ/face_landmark/blob/master/result_sample_0.jpg)

![result](https://github.com/JunrQ/face_landmark/blob/master/result_sample_1.jpg)

![result](https://github.com/JunrQ/face_landmark/blob/master/result_sample_2.jpg)

![result](https://github.com/JunrQ/face_landmark/blob/master/result_sample_3.jpg)