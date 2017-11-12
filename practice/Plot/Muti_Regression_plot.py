import Function_add_layer as exe5
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def plot_data(ax,x,y):
    ax.scatter(x, y)
    plt.ion()
    plt.show()
'''
parameter setting 
'''
learning_rate=0.1
iteration=100000

'''
Define Data (tensor)
'''
# Create a 1x300 float32 matrix
x_data=np.linspace(-1,1,300,np.float32)[:,np.newaxis]
# Create a 1x300 float32 matrix value is 0 from 0.05
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
# Create a 300x300 float32 matrix add noise to simulate the real condition
y_data=np.square(x_data)-0.5+noise
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot_data(ax,x_data,y_data)
'''
Define Architecture (flow(flowchart)) 
'''
# None mean Whatever number is what that it don't care about
xs=tf.placeholder(tf.float32,[None,1]);
ys=tf.placeholder(tf.float32,[None,1]);

L1=exe5.Add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=exe5.Add_layer(L1,10,1,activation_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
optim=tf.train.GradientDescentOptimizer(learning_rate)
train=optim.minimize(loss)
# initial this Architecture
init=tf.global_variables_initializer()

'''
Define Activaty
'''
with tf.Session() as sess:
    sess.run(init)
    loss_L=sess.run(loss,feed_dict={xs:x_data,ys:y_data})
    #print sess.run(init)
    for i in range(iteration):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})

        if loss_L==sess.run(loss,feed_dict={xs:x_data,ys:y_data}):
            print "iteration:\t" +str(i) \
                  +"\nloss:\t" + str(loss_L)
            break
        loss_L = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        if i%10==0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value=sess.run(prediction,feed_dict={xs:x_data})
            lines=ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.1)
            #print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
