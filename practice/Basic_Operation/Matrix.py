import tensorflow as tf
# Architecture define
# Create Matrix data
M1=tf.constant([[3,3]])
M2=tf.constant( [[2]
               ,[2]])
# matrix algorithm
product=tf.matmul(M1,M2)
# Write Method 1 Show result using session

sess=tf.Session()
result=sess.run(product)
print result

sess.close()
'''
# Write Method 2 Show result using session
with tf.Session as sess:
    result=sess.run(product)
    print result
'''
'''
import numpy as np
print "1:reporter 2:Do report"
I=[]
for i in range(8):
    I.append(i)
A=np.random.choice(8,8,replace=False)
A=A%2+1;
print str(I)+"\n"+str(A)
'''