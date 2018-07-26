""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)
print("the shape of whole data is ", data.shape)
print("The size of samples is ", n_samples)
#exit()

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
X = tf.placeholder(dtype=tf.float32, name='x')
Y = tf.placeholder(dtype=tf.float32, name='y')
#############################
########## TO DO ############
#############################

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w = tf.get_variable(name='weight', dtype=tf.float32, initializer=tf.zeros([1]))
b = tf.get_variable(name='bias', dtype=tf.float32, initializer=tf.zeros([1]))
#############################
########## TO DO ############
#############################

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = w * X + b 
#############################
########## TO DO ############
#############################

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name = 'loss')
#############################
########## TO DO ############
#############################

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

# Create a filewriter to write the model's graph to TensorBoard
#############################
########## TO DO ############
#############################

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    #############################
    ########## TO DO ############
    #############################
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graphs/linreg_starter', sess.graph)

    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            # Execute train_op and get the value of loss.
            # Don't forget to feed in data for placeholders
            _, e_loss = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += e_loss
            #print('x:{0}, y:{1}'.format(x, y))

        print('Average loss on epoch {0}: {1} '.format(i, total_loss/n_samples))
        #print('epoch{0} : w -> {1}, b -> {2} \n'.format(i, w.eval(), b.eval()))
   
    # close the writer when you're done using it
    #############################
    ########## TO DO ############
    #############################
    writer.close()
    
    # Step 9: output the values of w and b
    #############################
    ########## TO DO ############
    #############################
    w_out, b_out = w.eval(), b.eval()
    #w_out, b_out = sess.run([w, b])
    #with tf.variable_scope("", reuse = tf.AUTO_REUSE):
    #    w_out = sess.run(tf.get_variable("weight"))
    print('\nFinal w : {0} , b : {1}'.format(w_out, b_out))
    
print('Took: %f seconds' %(time.time() - start))

# uncomment the following lines to see the plot 
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
