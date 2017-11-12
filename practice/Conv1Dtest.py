import tensorflow as tf
import numpy as np
def display(tensor):
    print (tensor.shape)
    print (tensor)
batch=5
AP_num=3
window_size=3
In=np.ones([batch,window_size,AP_num]);
#In=np.array([[[7,8,9],[4,5,6],[1,2,3]]]);
#In=np.array([[[7,8,9]]]);
#In=np.array([[[1,1,1]]]);
print (In.shape)
X=tf.placeholder(tf.float32,[None,window_size,AP_num])
#[[1,1,1],[1,1,1],[1,1,1]],
#Y=tf.constant([[[1],[2],[3]]],dtype=tf.float32)
Y=tf.constant([[[1,1],[2,2],[3,3]],[[3,3],[2,2],[1,1]]],dtype=tf.float32)
#Y=tf.constant([[[1,1,1],[2,2,2],[3,3,3]]],dtype=tf.float32)
#Y=tf.constant([1,3,5,9,2,7,6],dtype=tf.float32)


print ('X')
display(X)

print ('Y')
display(Y)
out=tf.nn.conv1d(X,Y,stride=1,padding='VALID')
print ('out')
display(out)
result=tf.Session().run([out],feed_dict={X:In})
print ('Result')
print np.array(result).shape
print result

