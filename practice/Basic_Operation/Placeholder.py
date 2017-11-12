import tensorflow as tf
# define data type
v1=tf.constant([[5,2]])
v2=tf.constant([[5],[2]])
output=tf.matmul(v1,v2)
print v1
print v2
sess=tf.Session()
print(sess.run(output))
sess.close()