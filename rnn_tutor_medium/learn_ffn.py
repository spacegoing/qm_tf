import numpy as np
import tensorflow as tf

def generateData():
    x = np.linspace(-np.pi, np.pi, 21)
    y = np.sin(x)
    return (x, y)

sess = tf.InteractiveSession()

# a batch of inputs of 2 value each
inputs = tf.placeholder(tf.float32, shape=[None, 1])

# a batch of output of 1 value each
desired_outputs = tf.placeholder(tf.float32, shape=[None, 1])

# [!] define the number of hidden units in the first layer
hidden_units_1 = 4 

# connect 2 inputs to 3 hidden units
# [!] Initialize weights with random numbers, to make the network learn
weights_1 = tf.Variable(tf.truncated_normal([1, hidden_units_1]))

# [!] The biases are single values per hidden unit
biases_1 = tf.Variable(tf.zeros([hidden_units_1]))

# connect 2 inputs to every hidden unit. Add bias
layer_1_outputs = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)

# [!] The XOR problem is that the function is not linearly separable
# [!] A MLP (Multi layer perceptron) can learn to separe non linearly separable points ( you can
# think that it will learn hypercurves, not only hyperplanes)
# [!] Lets' add a new layer and change the layer 2 to output more than 1 value

# connect first hidden units to 2 hidden units in the second hidden layer
hidden_units_2 = 2
weights_2 = tf.Variable(tf.truncated_normal([hidden_units_1, hidden_units_2]))
# [!] The same of above
biases_2 = tf.Variable(tf.zeros([hidden_units_2]))

# connect the hidden units to the second hidden layer
layer_2_outputs = tf.nn.sigmoid(
    tf.matmul(layer_1_outputs, weights_2) + biases_2)

# [!] create the new layer
weights_3 = tf.Variable(tf.truncated_normal([hidden_units_2, 1]))
biases_3 = tf.Variable(tf.zeros([1]))
coe_3 = tf.Variable(tf.zeros([1]))

logits = coe_3 * tf.nn.sigmoid(tf.matmul(layer_2_outputs, weights_3) + biases_3)

# [!] The error function chosen is good for a multiclass classification taks, not for a XOR.
error_function = 0.5  tf.reduce_sum(tf.subtract(logits, desired_outputs)  tf.subtract(logits, desired_outputs))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(error_function)

sess.run(tf.initialize_all_variables())

x, y = generateData()

training_inputs = [[item] for item in x]

training_outputs = [[item] for item in y]

for i in range(5000):
    _, loss = sess.run([train_step, error_function],
                       feed_dict={inputs: np.array(training_inputs),
                                  desired_outputs: np.array(training_outputs)})
    print(loss)

print(sess.run(logits, feed_dict={inputs: np.array([[0]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[np.pi/4]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[np.pi/2]])}))
print(sess.run(logits, feed_dict={inputs: np.array([[np.pi/4*3]])}))

