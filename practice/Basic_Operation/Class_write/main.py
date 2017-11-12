from sympy import true
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

from CNN_number import CNN_number

#input=input_data.read_data_sets("MNIST_data",one_hot=true)

#CNN=CNN_number(inputdata=input)
#CNN.Do_model()
#CNN.run()
print tf.__version__
X=tf.constant([[1,0],[0,1]])
t=tf.constant([2])
Y=X*t
print tf.Session().run(Y)
#print tf.ops.mul(,)