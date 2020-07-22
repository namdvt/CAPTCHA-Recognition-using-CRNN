import glob
import os
import shutil
import re
import random


def write_annotation(root, file):
    f = open(root + '/' + file, 'w+')
    for image in glob.glob(root + '/*/*.jpg'):
        label = re.split('.jpg', image, flags=re.IGNORECASE)[0] + '.txt\n'
        f.write(image + '\t' + label)
    f.close()


def split_train_val_test(root):
    files = []

    for ext in ('*.png', '*.jpg'):
        files.extend(glob.glob(os.path.join(root, ext)))
    random.shuffle(files)

    for folder in ['/train', '/val', '/test']:
        if not os.path.exists(root + folder):
            os.mkdir(root + folder)

    for file in files[0:int(len(files) * 0.8)]:
        shutil.move(file, root + '/train/')

    for file in files[int(len(files) * 0.8):int(len(files) * 0.9)]:
        shutil.move(file, root + '/val/')

    for file in files[int(len(files) * 0.9):len(files)]:
        shutil.move(file, root + '/test')


if __name__ == '__main__':
    split_train_val_test('data/CAPTCHA Images')

    print()
