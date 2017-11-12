from sympy import true

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
def weight_v(shape):
    ini=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(ini)
def bias_v(shape):
    ini=tf.constant(0.1,shape=shape)
    return tf.Variable(ini)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pooling2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

'''
tensor
'''
lr=1e-4
iter=1000
input=input_data.read_data_sets('MNIST_data',one_hot=true)
print "Load Tensor\n"

'''
flow
'''
# Input
In=tf.placeholder(tf.float32,[None,784])/255
# Output
label=tf.placeholder(tf.float32,[None,10])
X_I=tf.reshape(In,[-1,28,28,1])
# Feature extraction Conv and pooling
# 28x28-28x28->14x14-14x14->7x7
patch_x=[5,5]
patch_y=[5,5]
bias=[32,64]
feature_in=[1,32]
feature_out=[32,64]
h_c=[]
h_p=[]
for i in range(0,len(bias)):
    # Convolution
    W_c = weight_v([patch_x[i],patch_y[i], feature_in[i], feature_out[i]])
    B_c = bias_v([bias[i]])
    if i==0:
        h_c.append(tf.nn.relu(conv2d(X_I,W_c)+B_c))
        h_p.append(max_pooling2x2(h_c[i]))
    else:
        h_c.append(tf.nn.relu(conv2d(h_p[i-1],W_c)+B_c))
        h_p.append(max_pooling2x2(h_c[i]))
feature_v=tf.reshape(h_p[len(h_p)-1],[-1,7*7*64])
print "Load Feature extration architect"+str(len(bias))+"\n"
# NN
W_fc=[7*7*64,1024,10]
b_fc=[1024,10]
h_fc=[]
h_fD=[]

# Drop out
Drop_out=tf.placeholder(tf.float32)

for i in range(0,len(b_fc)):
    wfc=weight_v([W_fc[i],W_fc[i+1]])
    bc=bias_v([b_fc[i]])
    if i==0:
        h_fc.append(tf.nn.relu(tf.matmul(feature_v, wfc) + bc))
        h_fD.append(tf.nn.dropout(h_fc[i],Drop_out))
    elif i==len(b_fc)-1:
        prediction = tf.nn.softmax(tf.matmul(h_fD[i-1],wfc)+bc)
    else:
        h_fc.append(tf.nn.relu(tf.matmul(h_fD[i - 1], wfc) + bc))
        h_fD.append(tf.nn.dropout(h_fc[i], Drop_out))

loss=tf.reduce_mean(-tf.reduce_sum(label*tf.log(prediction),reduction_indices=[1]))
train=tf.train.AdamOptimizer(lr).minimize(loss)
init=tf.global_variables_initializer()
print "Load NN architect"+str(len(b_fc))+"\n"
print "Begin RUN\n"
with tf.Session() as sess:
    batchx,batchy=input.train.next_batch(100)
    sess.run(init,feed_dict={In:batchx,label:batchy,Drop_out:0.5})
    # savepath=sever.save(sess,"M_NET/save_net.ckpt")
    for i in range(iter):
        sess.run(train,feed_dict={In:batchx,label:batchy,Drop_out:0.5})
        if i%10==1 and i!=0:
            pred=sess.run(prediction,feed_dict={In:input.test.images,Drop_out:1})
            ACC=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(pred,1),tf.arg_max(input.test.labels,1)),tf.float32))
            R=sess.run(ACC,feed_dict={In:input.test.images,label:input.test.labels,Drop_out:1})
            print str(i)+"\tacc:"+str(R)+"\n"