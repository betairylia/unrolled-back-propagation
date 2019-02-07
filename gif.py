from PIL import Image
import sys
import os

images = []
cnt = 0

for file in os.listdir(sys.argv[1]):
    if file[-3:] == 'png':
        im = Image.open(os.path.join(sys.argv[1], file))
        if cnt == 0:
            im_tmp = os.path.join(sys.argv[1], file)
        else:
            images.append(im)
        cnt += 1

im_tmp = Image.open(im_tmp)
im_tmp.save('output.gif', save_all = True, append_images = images, loop = 1, duration = 1)
