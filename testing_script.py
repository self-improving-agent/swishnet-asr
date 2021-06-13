import tensorflow as tf
import numpy as np

from model import MySwishNet

model_name = "my_swish_net_two_sec"
dataset = 'two_sec_3'
num_classes = 3

#device_name = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
device_name = '/cpu:0'

# Load datasets
test_x = np.load("processed_datasets/{}_test_x.npy".format(dataset), allow_pickle=True)
test_y = np.load("processed_datasets/{}_test_y.npy".format(dataset), allow_pickle=True)
test_y = np.eye(num_classes)[test_y.astype('int')]

valid_x = np.load("processed_datasets/{}_valid_x.npy".format(dataset), allow_pickle=True)
valid_y = np.load("processed_datasets/{}_valid_y.npy".format(dataset), allow_pickle=True)
valid_y = np.eye(num_classes)[valid_y.astype('int')]

# Define variables and operations
X = tf.compat.v1.placeholder(tf.float32, [None, test_x[0].shape[0], test_x[0].shape[1]])
Y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])

with tf.device(device_name):
	output = MySwishNet(X, input_shape=(test_x[0].shape[0],test_x[0].shape[1]), classes=num_classes)
	prediction = tf.nn.softmax(output)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.compat.v1.train.Saver()

# Run session
with tf.compat.v1.Session(config=tf.ConfigProto()) as sess:
	# Load model
	saver.restore(sess, "saved_models/{}/{}-120".format(dataset,model_name))

	# Evaluate validation and tes sets
	valid_acc = sess.run(accuracy, feed_dict={X : valid_x, Y : valid_y})
	test_acc = sess.run(accuracy, feed_dict={X : test_x, Y : test_y})

# Print results
print("Valid Accuracy: {:.4f}".format(valid_acc))
print("Test Accuracy: {:.4f}".format(test_acc))