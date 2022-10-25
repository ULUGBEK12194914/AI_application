#week6 Linear Regression
https://github.com/ULUGBEK12194914/AI_application/blob/main/LinearRegression(week6).ipynb

#week6 Session2
https://github.com/ULUGBEK12194914/AI_application/blob/main/week6(2session).ipynb

#week7e1(This code example shows how to access the CIFAR-10 dataset and display the image with index 100)
https://github.com/ULUGBEK12194914/AI_application/blob/main/week7e1.ipynb

#week7e2(This code example demonstrates how to use a convolutional neural network to solve an image classification problem, using the CIFAR-10 dataset.)
https://github.com/ULUGBEK12194914/AI_application/blob/main/week7e2.ipynb

#week7e3(This code example is very similar to week7e2 but the network is modified to use a more complex network)
https://github.com/ULUGBEK12194914/AI_application/blob/main/week7e3.ipynb

#week9e1(This code example demonstrates how to use a neural network to solve a regression problem, using the Boston housing dataset)
https://github.com/ULUGBEK12194914/AI_application/blob/main/week9e1.ipynb


# Linear Regression using TensorFlow

```bash
# First install the tensorFlow version 1.15.5
---------------------------------------------
pip install tensorflow==1.15.5

#import the libraries
----------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_parameter = 0.01
epochs = 300

sample_points = 50
x_train = np.linspace(0, 30, sample_points)
y_train = 6*x_train + 7*np.random.randn(sample_points)

# Noisy dataset
plt.plot(x_train, y_train, 'o')
# Noise free dataset 
plt.plot(x_train, 6*x_train)
plt.show()

Y = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = 'weights')
B = tf.Variable(np.random.randn(), name = 'bias')

#Create the model for regression
prediction = W*X + B

# Cost function
cost_iteration = tf.reduce_sum((prediction-Y)**2)/(2*sample_points)

#Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_parameter).minimize(cost_iteration)

# Initialize the variables
init = tf.global_variables_initializer()

# Start the Tensorflow session to execute the graph
---------------------------------------------------
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
      sess.run(optimizer, feed_dict = {X : x, Y : y})
    if not epoch%40:
      W1 = sess.run(W)
      B1 = sess.run(B)
      cost_iter = sess.run(cost_iteration, feed_dict = {X : x, Y : y})
      print('Epochs %f Cost %f Weight %f Bias %f' %(epoch, cost_iter, W1, B1))
  Weight = sess.run(W)
  Bias = sess.run(B)

  plt.plot(x_train, y_train, 'o')
  plt.plot(x_train,Weight*x_train+Bias)
  plt.show()
  
#output
-------
Epochs 0.000000 Cost 0.154752 Weight 5.665872 Bias 0.810562
Epochs 40.000000 Cost 0.389723 Weight 6.009754 Bias 0.670748
Epochs 80.000000 Cost 0.392550 Weight 6.015407 Bias 0.523763
Epochs 120.000000 Cost 0.395118 Weight 6.020525 Bias 0.390661
Epochs 160.000000 Cost 0.397452 Weight 6.025161 Bias 0.270135
Epochs 200.000000 Cost 0.399572 Weight 6.029358 Bias 0.160994
Epochs 240.000000 Cost 0.401493 Weight 6.033159 Bias 0.062164
Epochs 280.000000 Cost 0.403241 Weight 6.036601 Bias -0.027329
```
![](https://github.com/ULUGBEK12194914/AI_application/blob/main/visualization(screenshots)/download%20(1).png)
#Visualization of Linear Regression parameters using TensorFlow
```bash  
# Create the model for regression
with tf.name_scope("Model") as scope:
  prediction = W*X + B

# Add summary to study behaviour of weights and biases with epochs
weight_histogram = tf.summary.histogram("Weights", W)
bias_histogram = tf.summary.histogram("Bias", B)

# Cost function
with tf.name_scope("Cost_function") as scope:
  cost_iteration = tf.reduce_sum((prediction-Y)**2)/(2*sample_points)

# Record the scalar summary of the cost function
cost_summary = tf.summary.scalar("Cost", cost_iteration)

#Define the optimizer
with tf.name_scope("Training") as scope:
  optimizer = tf.train.GradientDescentOptimizer(learning_parameter).minimize(cost_iteration)

# Initialize the variables
init = tf.global_variables_initializer()

#Merge all the summaries into a single operator
merged_summary = tf.summary.merge_all()

# Define the tensorflow session
with tf.Session() as sess:
  sess.run(init)
  writer = tf.summary.FileWriter('./log', sess.graph)
  for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
      sess.run(optimizer, feed_dict = {X : x, Y : y})

      # Write logs for each epochs
      summary_epochs = sess.run(merged_summary, feed_dict = {X : x, Y : y})
      writer.add_summary(summary_epochs, epoch)
    if not epoch%40:
      W1 = sess.run(W)
      B1 = sess.run(B)
      cost_iter = sess.run(cost_iteration, feed_dict = {X : x, Y : y})
      print('Epochs %f Cost %f Weight %f Bias %f' %(epoch, cost_iter, W1, B1))
  Weight = sess.run(W)
  Bias = sess.run(B)

  plt.plot(x_train, y_train, 'o')
  plt.plot(x_train,Weight*x_train+Bias)
  plt.show()
  
#output
-------
Epochs 0.000000 Cost 0.154752 Weight 5.665872 Bias 0.810562
Epochs 40.000000 Cost 0.389723 Weight 6.009754 Bias 0.670748
Epochs 80.000000 Cost 0.392550 Weight 6.015407 Bias 0.523763
Epochs 120.000000 Cost 0.395118 Weight 6.020525 Bias 0.390661
Epochs 160.000000 Cost 0.397452 Weight 6.025161 Bias 0.270135
Epochs 200.000000 Cost 0.399572 Weight 6.029358 Bias 0.160994
Epochs 240.000000 Cost 0.401493 Weight 6.033159 Bias 0.062164
Epochs 280.000000 Cost 0.403241 Weight 6.036601 Bias -0.027329
```
![](https://github.com/ULUGBEK12194914/AI_application/blob/main/visualization(screenshots)/download.png)
#Tensorboard visualization
![](https://github.com/ULUGBEK12194914/AI_application/blob/main/visualization(screenshots)/Screen%20Shot%202022-10-04%20at%2015.28.26.png)


