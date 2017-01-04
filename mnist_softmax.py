# Download and read MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# Implementing Regression

# Import tensorflow
import tensorflow as tf

IMAGE_SIZE = 28 * 28
CLASS_SIZE = 10  # 0-9
# Create placeholder for MNIST image data
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

# Create Weights and Bias as a Variable
# Since these will be trained, it doesn't matter their initial values.
W = tf.Variable(tf.zeros([IMAGE_SIZE, CLASS_SIZE]))
b = tf.Variable(tf.zeros([CLASS_SIZE]))

# Define model
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Training

# Create placeholder for correct answers
y_ = tf.placeholder(tf.float32, [None, CLASS_SIZE])

# Define cross-entropy function
cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(tf.mul(y_, tf.log(y)), reduction_indices=[1]))

# Backpropaation
LEARNING_RATE = 0.5
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Initialize variables created
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Evaluate model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
