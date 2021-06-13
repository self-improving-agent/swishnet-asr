import math
import tensorflow as tf
import numpy as np

from model import MySwishNet


# Hyperparameters
learning_rate = 0.001
batch_size = 200
epochs = 120

# Setup
np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)

print_freq = 100
model_name = "my_swish_net_two_sec"
dataset = 'two_sec_3'
num_classes = 3

#device_name = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
device_name = '/cpu:0'

# Load datasets
train_x = np.load("processed_datasets/{}_train_x.npy".format(dataset), allow_pickle=True)
train_y = np.load("processed_datasets/{}_train_y.npy".format(dataset), allow_pickle=True)
train_y = np.eye(num_classes)[train_y.astype('int')]

valid_x = np.load("processed_datasets/{}_valid_x.npy".format(dataset), allow_pickle=True)
valid_y = np.load("processed_datasets/{}_valid_y.npy".format(dataset), allow_pickle=True)
valid_y = np.eye(num_classes)[valid_y.astype('int')]

# Define variables and operations
X = tf.compat.v1.placeholder(tf.float32, [None, train_x[0].shape[0], train_x[0].shape[1]])
Y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])

step = tf.Variable(0, trainable=False)

with tf.device(device_name):
	output = MySwishNet(X, input_shape=(train_x[0].shape[0],train_x[0].shape[1]), classes=num_classes)
	prediction = tf.nn.softmax(output)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
	
	# Cosine annealing with warm restarts
	learning_rate_decayed = tf.compat.v1.train.cosine_decay_restarts(learning_rate, step, 1000)
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_decayed)
	train_op = optimizer.minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.compat.v1.train.Saver()
init = tf.compat.v1.global_variables_initializer()

# Run session
with tf.compat.v1.Session(config=tf.ConfigProto()) as sess:
	sess.run(init)

	for e in range(epochs):	
		# Shuffle
		shuffle_order = np.random.permutation(len(train_x))
		train_x = train_x[shuffle_order]
		train_y = train_y[shuffle_order]

		# Split to batches
		batches_x = np.array_split(train_x, math.ceil(len(train_x) / batch_size))
		batches_y = np.array_split(train_y, math.ceil(len(train_y) / batch_size))

		# Process batches
		for b in range(len(batches_x)):
			sess.run(train_op, feed_dict={X : batches_x[b], Y : batches_y[b]})

			if b % print_freq == 0:
				loss, acc = sess.run([loss_op, accuracy], feed_dict={X : batches_x[b], Y : batches_y[b]})

				print("Batch: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}".format(b+1, len(batches_x), loss, acc))

		# Evaluate validation set
		valid_loss, valid_acc = sess.run([loss_op, accuracy], feed_dict={X : valid_x, Y : valid_y})
		print("Epoch: {}/{}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}".format(e+1, epochs, valid_loss, valid_acc))

		# Save model
		saver.save(sess, "saved_models/{}/{}".format(dataset, model_name), global_step=e+1)