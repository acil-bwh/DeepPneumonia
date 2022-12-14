import cv2
import numpy as np
from skimage import measure
from scipy import ndimage


def recolor_resize(img, pix=256):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('', end = '')
    img = cv2.resize(img, (pix, pix))
    img = np.expand_dims(img, axis=-1)
    return img


def recolor(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('', end = '')
    return img


def normalize(img):
    return (img - np.mean(img))/ np.std(img)


def remove_pieces(mask):
    mask = measure.label(mask)
    ntotal = {k: (k==mask).sum() for k in np.unique(mask) if k >0}
    k = list(ntotal.keys())[np.argmax(list(ntotal.values()))]
    mask = k==mask
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((5,5)))
    return mask


def des_normalize(img):
    return cv2.normalize(img, None, alpha = 0, beta = 255,
                         norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_16UC1)


def apply_mask(img, model):
    pix1 = img.shape[0]
    pix2 = img.shape[1]
    # Transform image into grayscale
    img = recolor(img)
    # New image with mask model input size
    img_2 = normalize(recolor_resize(img, 256))[np.newaxis,...]
    # Mask generation
    mask = model.predict(img_2, verbose = 0)[0,...]
    # Resize mask to the original image size
    mask = cv2.resize(mask, (pix2, pix1))
    # Clean the mask
    mask = remove_pieces(mask > 0.5)
    # Apply the mask over the image
    return img*mask




