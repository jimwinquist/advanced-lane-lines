import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import image_process

# TODO: Refactor to reduce duplication and create single flexible API with a common interface
# TODO: Add documentation to doc strings for each function
#------------------------------------------------------------------------------
def batch_gray_binary(thresh_min, thresh_max, display=False):
    '''

    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        gray = image_process.grayscale(image)
        gray_binary = np.zeros_like(gray)
        gray_binary[(gray > thresh_min) & (gray <= thresh_max)] = 1
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (gray_binary*255).astype('uint8')
        cv2.imwrite('output_images/gray_thresholds/{0}_gray_thresh_{1}_{2}.png'.format(base, thresh_min, thresh_max), output)

        if display:
            image_process.showImage('Gray Binary Threshold', output)

#------------------------------------------------------------------------------
def batch_rgb_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        ch = image[:,:,channel].copy()
        rgb_binary = np.zeros_like(ch)
        rgb_binary[(ch > thresh_min) & (ch <= thresh_max)] = 1
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (rgb_binary*255).astype('uint8')
        cv2.imwrite('output_images/rgb_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

        if display:
            image_process.showImage('RGB Binary Threshold', output)

#------------------------------------------------------------------------------
def batch_hsv_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        hsv_binary = image_process.hsvBinary(image, channel, (thresh_min, thresh_max), display=display)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (hsv_binary*255).astype('uint8')
        cv2.imwrite('output_images/hsv_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_hls_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        hls_binary = image_process.hlsBinary(image, channel, (thresh_min, thresh_max), display=display)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (hls_binary*255).astype('uint8')
        cv2.imwrite('output_images/hls_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_lab_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        lab_binary = image_process.labBinary(image, channel, (thresh_min, thresh_max), display=display)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (lab_binary*255).astype('uint8')
        cv2.imwrite('output_images/lab_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_luv_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        luv_binary = image_process.luvBinary(image, channel, (thresh_min, thresh_max), display=display)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (luv_binary*255).astype('uint8')
        cv2.imwrite('output_images/luv_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_ycrcb_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        ycrcb_binary = image_process.ycbcrBinary(image, channel, (thresh_min, thresh_max), display=display)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (ycrcb_binary*255).astype('uint8')
        cv2.imwrite('output_images/ycrcb_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_yuv_binary(channel, thresh_min, thresh_max, display=False):
    '''

    :param channel:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        yuv_binary = image_process.yuvBinary(image, channel, (thresh_min, thresh_max), display=display)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        output = (yuv_binary*255).astype('uint8')
        cv2.imwrite('output_images/yuv_thresholds/{0}_ch{1}_thresh_{2}_{3}.png'.format(base, channel, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_sobel_binary(orient, thresh_min, thresh_max, display=False):
    '''

    :param orient:
    :param thresh_min:
    :param thresh_max:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        gray = image_process.grayscale(image)
        if orient == 'x':
            output = (image_process.absSobelThreshold(gray, orient='x', thresh=(thresh_min, thresh_max)) * 255).astype('uint8')
        if orient == 'y':
            output = (image_process.absSobelThreshold(gray, orient='y', thresh=(thresh_min, thresh_max)) * 255).astype('uint8')
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        cv2.imwrite('output_images/sobel_thresholds/{0}_sobel{1}_thresh_{2}_{3}.png'.format(base, orient, thresh_min, thresh_max), output)

#------------------------------------------------------------------------------
def batch_adaptive_binary(kernel_size, display=False):
    '''

    :param kernel_size:
    :param display:
    :return:
    '''
    for imagePath in glob.glob('test_images/*.jpg'):
        image = image_process.readImage(imagePath)
        gray = image_process.grayscale(image)
        blurred = cv2.medianBlur(gray, kernel_size)
        adaptive_binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,7,2)
        base, ext = os.path.splitext(imagePath)
        base = os.path.basename(base)
        cv2.imwrite('output_images/adaptive_thresholds/{0}_adaptive_thresh.png'.format(base), adaptive_binary)

        if display:
            image_process.showImage('Adaptive Binary Threshold', adaptive_binary)

#------------------------------------------------------------------------------
def plot_images(title1, img1, title2, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()