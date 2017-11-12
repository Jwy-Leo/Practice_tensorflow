import tensorflow as tf
def Add_layer(inputs,in_size,out_size,activation_function=None):

    # deffierent from truncate function
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))

    # tunring bias to 0.1
    # the default bias isn't command assign as 0
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weight)+biases
    if activation_function==None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
