import numpy as np
import tensorflow as tf

a = np.zeros([3, 3, 3, 3])
a[1, 1, :, :] = 0.25    
#filter format is HWCN, that is to change value in posi(1, 1) to 0.25 for all C 、N dimentions.
a[0, 1, :, :] = 0.125
a[1, 0, :, :] = 0.125
a[2, 1, :, :] = 0.125
a[1, 2, :, :] = 0.125
a[0, 0, :, :] = 0.0625
a[0, 2, :, :] = 0.0625
a[2, 0, :, :] = 0.0625
a[2, 2, :, :] = 0.0625

BLUR_FILTER_RGB = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 1, 1])
# a[1, 1, :, :] = 0.25
# a[0, 1, :, :] = 0.125
# a[1, 0, :, :] = 0.125
# a[2, 1, :, :] = 0.125
# a[1, 2, :, :] = 0.125
# a[0, 0, :, :] = 0.0625
# a[0, 2, :, :] = 0.0625
# a[2, 0, :, :] = 0.0625
# a[2, 2, :, :] = 0.0625
a[1, 1, :, :] = 1.0
a[0, 1, :, :] = 1.0
a[1, 0, :, :] = 1.0
a[2, 1, :, :] = 1.0
a[1, 2, :, :] = 1.0
a[0, 0, :, :] = 1.0
a[0, 2, :, :] = 1.0
a[2, 0, :, :] = 1.0
a[2, 2, :, :] = 1.0
BLUR_FILTER = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 3, 3])
a[1, 1, :, :] = 5
a[0, 1, :, :] = -1
a[1, 0, :, :] = -1
a[2, 1, :, :] = -1
a[1, 2, :, :] = -1

SHARPEN_FILTER_RGB = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 1, 1])
a[1, 1, :, :] = 5
a[0, 1, :, :] = -1
a[1, 0, :, :] = -1
a[2, 1, :, :] = -1
a[1, 2, :, :] = -1

SHARPEN_FILTER = tf.constant(a, dtype=tf.float32)

# a = np.zeros([3, 3, 3, 3])
# a[:, :, :, :] = -1
# a[1, 1, :, :] = 8

# EDGE_FILTER_RGB = tf.constant(a, dtype=tf.float32)

EDGE_FILTER_RGB = tf.constant([
			[[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]],
            [[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
			[[ 8., 0., 0.], [ 0., 8., 0.], [ 0., 0., 8.]],
			[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]],
			[[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
			[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
			[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]]
])

a = np.zeros([3, 3, 1, 1])
# a[:, :, :, :] = -1
# a[1, 1, :, :] = 8
a[0, 1, :, :] = -1
a[1, 0, :, :] = -1
a[1, 2, :, :] = -1
a[2, 1, :, :] = -1
a[1, 1, :, :] = 4

EDGE_FILTER = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 3, 3])
a[0, :, :, :] = 1
a[0, 1, :, :] = 2 # originally 2
a[2, :, :, :] = -1
a[2, 1, :, :] = -2

TOP_SOBEL_RGB = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 1, 1])
a[0, :, :, :] = 1
a[0, 1, :, :] = 2 # originally 2
a[2, :, :, :] = -1
a[2, 1, :, :] = -2

TOP_SOBEL = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 3, 3])
a[0, 0, :, :] = -2
a[0, 1, :, :] = -1 
a[1, 0, :, :] = -1
a[1, 1, :, :] = 1
a[1, 2, :, :] = 1
a[2, 1, :, :] = 1
a[2, 2, :, :] = 2

EMBOSS_FILTER_RGB = tf.constant(a, dtype=tf.float32)

a = np.zeros([3, 3, 1, 1])
a[0, 0, :, :] = -2
a[0, 1, :, :] = -1 
a[1, 0, :, :] = -1
a[1, 1, :, :] = 1
a[1, 2, :, :] = 1
a[2, 1, :, :] = 1
a[2, 2, :, :] = 2
EMBOSS_FILTER = tf.constant(a, dtype=tf.float32)
