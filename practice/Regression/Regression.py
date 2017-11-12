import tensorflow as tf
import numpy as np
# Architecture define
'''
Design parameter part
1. learning rate
2. iteration 
'''
# Learning rate
learn_rate=0.5
# iteration
iter=500
'''
Data part
'''
# DATA
X_D=np.random.rand(100).astype((np.float32))
# LABEL
Y_D=X_D*0.1+0.3
'''
Weight Bias initial
'''
# WEIGHT
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
# BIAS
Bias=tf.Variable(tf.zeros([1]))


# OUTPUT
y_O=W*X_D+Bias

'''
Design part 
1.loss
2.learning method
3.learning target
'''
# Loss function
loss=tf.reduce_mean(tf.square((y_O-Y_D)));
# OPT
optimizer=tf.train.GradientDescentOptimizer(learn_rate)

# Train mode
train=optimizer.minimize(loss)

'''
Initial the model
'''
init=tf.global_variables_initializer();
with tf.Session() as session:
    session.run(init)
    L_W = session.run(W);
    L_B = session.run(Bias);
    print "initial parameter" + str(session.run(W)) + str(session.run(Bias))
    for i in range(iter):
        session.run(train)
        if (L_W==session.run(W) and L_B==session.run(Bias)):
            print "Exit in iteration:\t"+str(i)+\
                  "\nlearning rate:\t\t"+str(learn_rate)+\
                  "\nMaximum iteration:\t"+str(iter)
            break
        L_W = session.run(W);
        L_B = session.run(Bias);
        if i % 20 == 0 and i!=1:
            print str(i)+str(session.run(W))+str(session.run(Bias))


