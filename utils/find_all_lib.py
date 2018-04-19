import os
import shutil

def move_lib(parent=r'E:\tensorflow-r1.3\tensorflow-r1.3\tensorflow\contrib\cmake\build',
             target_path=r'E:\tensorflow-r1.3\tensorflow-r1.3\tensorflow\contrib\cmake\build\all_lib',
             pattern='.lib'):
    for p in os.listdir(parent):
        tmp_path = os.path.join(parent, p)
        if os.path.isdir(tmp_path):
            move_lib(tmp_path)
        else:
            if p.endswith(pattern):
                shutil.copy(tmp_path, os.path.join(target_path, p))

if __name__ == '__main__':
    move_lib()
