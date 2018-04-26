import glob
import os
import re

def generate_file_list_txt(target_txt, parent_path, glob_par="*.jpg"):
    with open(target_txt, 'w') as f:
        for single_parent_path in parent_path:
            glob_res = glob.glob(os.path.join(single_parent_path, glob_par))
            glob_res = filter(lambda x:not os.path.basename(x).startswith('dlib_'), glob_res)
            f.write('\n'.join(glob_res))

if __name__ == '__main__':
    generate_file_list_txt('img_list_val.txt',
                           [r'E:\python_vanilla\validation_dataset\ce',
                            r'E:\python_vanilla\validation_dataset\di',
                            r'E:\python_vanilla\validation_dataset\glasses',
                            r'E:\python_vanilla\validation_dataset\hat',
                            r'E:\python_vanilla\validation_dataset\tai',
                            r'E:\python_vanilla\validation_dataset\zheng']
    )

    generate_file_list_txt('img_list.txt',
                           [r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected',
                            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\ce',
                            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\hu',
                            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\low',
                            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\21_points_1',
                            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\21_points_2',
                            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\21_points_3'],
                           '*_0.jpg'
                           )

