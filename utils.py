import numpy as np
import matplotlib.pyplot as plt

from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY

def import_img(f_name, bw=True, newshape=False):

    # Color convert
    if bw:
        img = cvtColor( imread(f_name), COLOR_BGR2GRAY)
    else:
        img = imread(f_name)

    # Resize
    if newshape != False:
        img = resize(img, newshape)

    print('Input size: ',img.shape)
    print(f"dtype: {img.dtype} | max: {np.max(img)} | min: {np.min(img)}")

    #plt.subplots(figsize=(10, 10))
    plt.imshow(img, cmap='gray')

    return img.astype(np.int32)
