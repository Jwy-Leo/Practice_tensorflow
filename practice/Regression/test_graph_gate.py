import Function_add_layer
import tensorflow as tf
import numpy as np
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

'''
Define Architecture (flow(flowchart)) 
'''
# None mean Whatever number is what that it don't care about
xs=tf.placeholder(tf.float32,[None,1]);
ys=tf.placeholder(tf.float32,[None,1]);

L1=Function_add_layer.Add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=Function_add_layer.Add_layer(L1,10,1,activation_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#loss2=tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])))
optim=tf.train.GradientDescentOptimizer(learning_rate)
train=optim.minimize(loss)
#O1=optim.compute_gradients(loss,gate_gradients=2)
#O2=optim.compute_gradients(loss2,gate_gradients=2)
'''
print (type(O1))
print (O1[0][1].value())
print (type(O2))
print (O2)
train1=optim.apply_gradients(O1)
train2=optim.apply_gradients(O2)
print (type(train1))
print (type(train2))
'''
#merged=tf.summary.merge_all()
FW=tf.summary.FileWriter('tensorboard/',tf.Session().graph)
#train=optim.minimize(loss)
# initial this Architecture
init=tf.global_variables_initializer()

'''
Define Activaity
concat([] 1)summary
pack->stack
'''
with tf.Session() as sess:
    init.run()
    loss_L=sess.run(loss,feed_dict={xs:x_data,ys:y_data})
    #print sess.run(init)
    for i in range(iteration):

        run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        loss_L=sess.run(train,feed_dict={xs:x_data,ys:y_data},options=run_option,run_metadata=run_metadata)
        
        if loss_L==sess.run(loss,feed_dict={xs:x_data,ys:y_data}):
            print "iteration:\t" +str(i) \
                  +"\nloss:\t" + str(loss_L)
            break
        loss_L = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        
        if i%50==0:
            FW.add_run_metadata(run_metadata, 'step%03d' % i)
            #FW.add_summary(summary,i)
            #print num2str(i)+'\t'
            print ('%3d\t'%i)
            tf.summary.tensor_summary()
            print sess.run(loss, feed_dict={xs: x_data, ys: y_data})
    FW.close