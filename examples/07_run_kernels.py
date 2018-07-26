"""
Simple examples of convolution to do some basic filters
Also demonstrates the use of TensorFlow data readers.

We will use some popular filters for our image.
It seems to be working with grayscale images, but not with rgb images.
It's probably because I didn't choose the right kernels for rgb images.

kernels for rgb images have dimensions 3 x 3 x 3 x 3
kernels for grayscale images have dimensions 3 x 3 x 1 x 1 （H,W,C,N)

CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
sys.path.append('..') #貌似是增加搜索路径

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import kernels

def read_one_image(filename):
    ''' This method is to show how to read image from a file into a tensor.
    The output is a tensor object.
    '''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded, tf.float32) / 256.0
    return image

def convolve(image, kernels, rgb=True, strides=[1, 3, 3, 1], padding='SAME'):
    images = [image[0]]  #从NHWC取第一个HWC。以此类推. 产生一个新的list，后续调用append函数增加list的元素
    for i, kernel in enumerate(kernels): #枚举，同时获得索引和值
        #conv2d返回一个与input同类型的tensor，这里是NHWC。[0]表示取第一个HWC
        filtered_image = tf.nn.conv2d(image, 
                                      kernel, 
                                      strides=strides,
                                      padding=padding)[0]
        if i == 2: # when i=2, kernal is TOP_SOBEL_FILTER. may exceed 255
            filtered_image = tf.minimum(tf.nn.relu(filtered_image), 255)
        images.append(filtered_image) #将处理过的图层在C方向增加
    return images

def show_images(images, rgb=True):
    gs = gridspec.GridSpec(1, len(images))  #define the shape of figure, eg.(2, 3), total 6 parts in a figure.
    for i, image in enumerate(images):      #similar to line 40. index counts from 0
        plt.subplot(gs[0, i])
        if rgb:
            plt.imshow(image)
        else: 
            image = image.reshape(image.shape[0], image.shape[1])
            plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()

def main():
    rgb = True
    if rgb:
    #kernels_list是一个list，每个元素都是4维的常量tensor。list没有shape、eval()属性
        kernels_list = [kernels.BLUR_FILTER_RGB, 
                        kernels.SHARPEN_FILTER_RGB, 
                        kernels.EDGE_FILTER_RGB,
                        kernels.TOP_SOBEL_RGB,
                        kernels.EMBOSS_FILTER_RGB]
    else:
        kernels_list = [kernels.BLUR_FILTER,
                        kernels.SHARPEN_FILTER,
                        kernels.EDGE_FILTER,
                        kernels.TOP_SOBEL,
                        kernels.EMBOSS_FILTER]

    kernels_list = kernels_list[1:] #切片操作，表示取除第一个元素之外的所有list内的元素并返回
    image = read_one_image('data/friday.jpg')   #image type is class <list>
    #print(np.shape(kernels_list))  -> (4, )
    
    if not rgb:
        image = tf.image.rgb_to_grayscale(image)

    image = tf.expand_dims(image, 0) #init shape:(28, 28, 3) -> after shape:(1, 28, 28, 3).but content will unchange
    images = convolve(image, kernels_list, rgb)
    
    with tf.Session() as sess:
        #print(image.shape)     -> unknow
        images = sess.run(images) # convert images from tensors to float values
        #print('kernal_list[0] shape : {0} \nContent :\n {1}'.format(kernels_list[0].shape, kernels_list[0].eval()))
    show_images(images, rgb)

if __name__ == '__main__':
    main()
