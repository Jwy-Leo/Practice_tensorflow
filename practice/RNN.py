import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
# parameters
learning_rate=0.001
train_iter=100000
batch_size=128
n_in=28
n_step=28
n_HU=128
n_classes=10
x=tf.placeholder(tf.float32,[None,n_step,n_in])
y=tf.placeholder(tf.float32,[None,n_classes])
weight={
    'in': tf.Variable(tf.random_normal([n_in,n_HU])),
    'out':tf.Variable(tf.random_normal([n_HU,n_classes]))
}
bias={
    'in': tf.Variable(tf.constant(0.1,shape=[n_HU,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}
def RNN(X,weight,bias):
    X=tf.reshape(X,[-1,n_in])
    X_in=tf.matmul(X,weight['in'])+bias['in']
    print(type(X_in))
    X_in=tf.reshape(X_in,[-1,n_step,n_HU])
    if int((tf.__version__).split('.')[1])<12 and int((tf.__version__).split('.')[0]<1):
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_HU,forget_bias=0.0,state_is_tupple=True)
    else:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_HU, forget_bias=0.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        #cell=tf.contrib.rnn();
        #cell=tf.nn.dynamic_rnn(n_HU,forget_bias=1.0,state_is_tupple=True)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    print ("OUT:\t" + str(outputs) + "\n")
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weight['out']) + bias['out']
    return results

pred = RNN(x, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < train_iter:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step,n_in])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys,}))
        step += 1
print 'FINISH\n'
'''