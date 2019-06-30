import tensorflow as tf

'''
本文档描述以下两个TensorFlow类
tf.Variable
tf.train.Saver
'''

# 创建
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

# 初始化
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()
# Later, when launching the model
with tf.Session() as sess:
    # Run the init operation.
    sess.run(init_op)
    # Use the model

# 由另一个变量初始化
'''
你有时候会需要用另一个变量的初始化值给当前变量初始化。
由于tf.initialize_all_variables()是并行地初始化所有变量，
所以在有这种需求的情况下需要小心。

用其它变量的值初始化一个新的变量时，
使用其它变量的initialized_value()属性。
你可以直接把已初始化的值作为新变量的初始值，
或者把它当做tensor计算得到一个值赋予新变量。
'''
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")

# 自定义初始化
'''
tf.initialize_all_variables()函数便捷地添加一个op
来初始化模型的所有变量。你也可以给它传入一组变量进行初始化。
'''

# 保存变量
# 用tf.train.Saver()创建一个Saver来管理模型中的所有变量。
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: ", save_path)

# 恢复变量
# 用同一个Saver对象来恢复变量。
# 注意，当你从文件中恢复变量时，不需要事先对它们做初始化。
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    # Do some work with the model

# 选择存储和恢复哪些变量,传入Python字典
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
