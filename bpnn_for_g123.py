import tensorflow as tf
import numpy as np


def batch(x,y,n):
    index = np.random.choice(6221, n, replace=False)
    return [[x[i] for i in index], [y[i] for i in index]]

'''
def computer_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
'''

train_x = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
train_y = np.loadtxt("train_y.csv", delimiter=",", skiprows=1)

xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 5])

Weights_1 = tf.Variable(tf.random_normal([4, 50])*0.01)
biases_1 = tf.Variable(tf.zeros([1, 50]) + 0.0001)
Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1

layer1_out = tf.nn.relu(Wx_plus_b_1)

Weights_2 = tf.Variable(tf.random_normal([50, 30])*0.01)
biases_2 = tf.Variable(tf.zeros([1, 30]) + 0.0001)
Wx_plus_b_2 = tf.matmul(layer1_out, Weights_2) + biases_2

layer2_out = tf.nn.relu(Wx_plus_b_2)

Weights_3 = tf.Variable(tf.random_normal([30, 5])*0.01)
biases_3 = tf.Variable(tf.zeros([1, 5]) + 0.0001)
prediction = tf.matmul(layer2_out, Weights_3) + biases_3

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.9, beta2=0.999).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(0, 3000):
    #[batch_x, batch_y] = batch(train_x, train_y, 300)
    #sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y})
    sess.run(train_step, feed_dict={xs:train_x, ys:train_y})
    if step % 500 == 0:
        print(step, sess.run(loss, feed_dict={xs:train_x, ys:train_y}))

'''
y_hat = np.dot((np.dot(train_x,o1)+o2),o3)+o4
print(y_hat)
'''
y_hat = sess.run(prediction,feed_dict={xs:train_x})

#np.savetxt('train_y_out.csv', train_y, delimiter = ',', fmt="%.6f")
#np.savetxt('y_hat.csv', y_hat, delimiter = ',', fmt="%.6f")