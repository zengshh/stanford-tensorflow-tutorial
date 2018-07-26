""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
""" 
This program reveal the whole procedure of building\training\validating one conv-net--mnist here.
Class mode provides a clear and reuseable way to achieve it but seems a little heavy. Maybe some methods can be merged to one. But at last, this is deserved reference.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf

import utils

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    #with the scope, initializing several conv_relu will produce different variables, even they seem to share the same name
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:   
        in_channels = inputs.shape[-1]      #get the C of input:(N, H, W, C)
        kernel = tf.get_variable('kernel',
                                [k_size, k_size, in_channels, filters], #(K, K, C, N)
                                initializer=tf.truncated_normal_initializer() )
        biases = tf.get_variable('biases',
                                [filters], # equal to [filters, ], produce a 1-dim vector
                                initializer=tf.random_normal_initializer() )
        conv = tf.nn.conv2d(inputs, kernel, strides = [1, stride, stride, 1], padding = padding)
    
    return tf.nn.relu(conv + biases, name = scope.name) # + uses broadcast method.

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:   
        pool = tf.nn.max_pool(inputs,
                            ksize = [1, ksize, ksize, 1],
                            strides = [1, stride, stride, 1],
                            padding = padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]       #get the last dime of inputs 
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

#the methods of class don't return values

class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)      #used for drop-out
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = True

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size) # without val_data
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()               #fetch batch-size samples
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])   # -1 represents not-specific
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data

    def inference(self):
        '''
        Build the model according to the description we've shown in class
        '''
        conv1 = conv_relu(inputs = self.img,
                            filters = 32,
                            k_size = 5,
                            stride = 1,
                            padding = 'SAME',
                            scope_name = 'conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')  # Pay attention to the position
        
        conv2 = conv_relu(inputs = pool1,
                            filters = 64,
                            k_size = 5,
                            stride = 1,
                            padding = 'SAME',
                            scope_name = 'conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')  # Pay attention to the position

        feature_dim = pool2.shape[1]*pool2.shape[2]*pool2.shape[3] # H*W*C
        pool2 = tf.reshape(pool2, [-1, feature_dim])     #flatten the NHWC into two dime. -1 should be pool2.shape[0]

        fc = fully_connected(pool2, 1024, 'fc')
        dropout = tf.nn.dropout(x = tf.nn.relu(fc), keep_prob = self.keep_prob, name = 'relu_dropout')

        self.logits = fully_connected(dropout, self.n_classes, 'logits')

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        tf.nn.softmax_cross_entropy_with_logits
        softmax is applied internally
        don't forget to compute mean cross all sample in a batch
        '''
        with tf.name_scope('loss'):
            #return 1-D tensor of length batch-size of the same type as logits
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        Don't forget to use global step
        '''
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,  global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        Remember to track both training loss and test accuracy
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()
 
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))   #wrong? should be reduce_mean?See later

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_starter/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step             #attentin! step was returned here!!

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_starter')
        writer = tf.summary.FileWriter('./graphs/convnet_starter', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=2)
            #the restore operation is not necessary here while training.
            #ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_starter/checkpoint'))
            #if ckpt and ckpt.model_checkpoint_path:
            #   saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=30)
