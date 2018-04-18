# prepare dataset
python ./prepare_dataset.py

/home/cuishi/caffe/.build_release/tools/caffe train --solver=./solver.prototxt  --gpu=0