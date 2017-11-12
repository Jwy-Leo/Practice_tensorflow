import tensorflow as tf
import numpy as np
N1=np.ones(shape=[3,3])
N2=np.ones(shape=[4,3])*4
C1=tf.constant(N1,tf.float32)
C2=tf.constant(N2,tf.float32)
TM=tf.sqrt(tf.reduce_sum(tf.square(C2),axis=1,keep_dims=True))
D=C2/TM
with tf.Session() as sess:
    print (sess.run(C1))
    print (sess.run(C2))
    print (sess.run(TM))
    print (sess.run(D))