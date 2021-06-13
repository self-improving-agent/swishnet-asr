import math
import tensorflow as tf
import numpy as np

from model import MySwishNet


np.random.seed(42)

learning_rate = 0.001
batch_size = 200
epochs = 50

print_freq = 100
model_name = "my_swish_net_two_sec"
num_classes = 2

#device_name = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
device_name = '/cpu:0'

# Load datasets
finetune_x = np.load("processed_datasets/gtzan_finetune_x.npy", allow_pickle=True)
finetune_y = np.load("processed_datasets/gtzan_finetune_y.npy", allow_pickle=True)
finetune_y = np.eye(num_classes)[finetune_y.astype('int')]

evaluate_x = np.load("processed_datasets/gtzan_evaluate_x.npy", allow_pickle=True)
evaluate_y = np.load("processed_datasets/gtzan_evaluate_y.npy", allow_pickle=True)
evaluate_y = np.eye(num_classes)[evaluate_y.astype('int')]

# Define variables and operations
X = tf.compat.v1.placeholder(tf.float32, [None, finetune_x[0].shape[0], finetune_x[0].shape[1]])
Y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])

with tf.device(device_name):
	output = MySwishNet(X, input_shape=(finetune_x[0].shape[0],finetune_x[0].shape[1]), classes=num_classes)
	prediction = tf.nn.softmax(output)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))

	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.compat.v1.train.Saver()

# Run session
with tf.compat.v1.Session(config=tf.ConfigProto()) as sess:
	# Load model
	saver.restore(sess, "saved_models/two_sec/{}-120".format(model_name))

	# Finetune
	for e in range(epochs):	
		# Shuffle
		shuffle_order = np.random.permutation(len(finetune_x))
		finetune_x = finetune_x[shuffle_order]
		finetune_y = finetune_y[shuffle_order]

		# Split to batches
		batches_x = np.array_split(finetune_x, math.ceil(len(finetune_x) / batch_size))
		batches_y = np.array_split(finetune_y, math.ceil(len(finetune_y) / batch_size))

		# Process batches
		for b in range(len(batches_x)):
			sess.run(train_op, feed_dict={X : batches_x[b], Y : batches_y[b]})

			if b % print_freq == 0:
				loss, acc = sess.run([loss_op, accuracy], feed_dict={X : batches_x[b], Y : batches_y[b]})

				print("Batch: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}".format(b+1, len(batches_x), loss, acc))

		evaluate_loss, evaluate_acc = sess.run([loss_op, accuracy], feed_dict={X : evaluate_x, Y : evaluate_y})
		print("Epoch: {}/{}, Evaluate Loss: {:.4f}, Evaluate Accuracy: {:.4f}".format(e+1, epochs, evaluate_loss, evaluate_acc))

	print("Finetuning finished")

	# Evaluate
	evaluate_acc = sess.run(accuracy, feed_dict={X : evaluate_x, Y : evaluate_y})
	saver.save(sess, "saved_models/gtzan_finetuned")

# Print result
print("GTZAN Accuracy: {:.4f}".format(evaluate_acc))