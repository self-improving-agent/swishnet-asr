import sys
import numpy as np
import tensorflow as tf


def causal_conv1D(x, filters, length, stride=1):
	x_input = tf.compat.v1.layers.Conv1D(filters=filters,
									 kernel_size=length,
									 dilation_rate=stride,
									 strides=1,
									 padding="causal")(x)
	x_sigmoid = tf.math.sigmoid(x_input)
	x_tanh = tf.math.tanh(x_input)
	x_output = tf.math.multiply(x_sigmoid, x_tanh)

	# Uncomment for separated version
	# x_input1 = tf.layers.Conv1D(filters=filters // 2,
	# 								 kernel_size=length,
	# 								 dilation_rate=stride,
	# 								 strides=1,
	# 								 padding="causal")(x)
	# x_sigmoid = tf.math.sigmoid(x_input1)

	# x_input2 = tf.layers.Conv1D(filters=filters // 2,
	# 								 kernel_size=length,
	# 								 dilation_rate=stride,
	# 								 strides=1,
	# 								 padding="causal")(x)
	# x_tanh = tf.math.tanh(x_input2)

	# x_output = tf.math.multiply(x_sigmoid, x_tanh)

	return x_output


def MySwishNet(X, input_shape, classes):
	x_input = X

	# Block 1
	x_up = causal_conv1D(x_input, filters=16, length=3)
	x_down = causal_conv1D(x_input, filters=16, length=6)
	x = tf.concat([x_up, x_down], axis=-1)

	# Block 2
	x_up = causal_conv1D(x, filters=8, length=3)
	x_down = causal_conv1D(x, filters=8, length=6)
	x_skip1 = tf.concat([x_up, x_down], axis=-1)

	# Block 3
	x_up = causal_conv1D(x_skip1, filters=8, length=3)
	x_down = causal_conv1D(x_skip1, filters=8, length=6)
	x = tf.concat([x_up, x_down], axis=-1)

	x = tf.math.add(x_skip1, x)

	# Block 4
	x_skip2 = causal_conv1D(x, filters=16, length=3, stride=3)
	x = tf.math.add(x, x_skip2)

	# Block 5
	x_skip3 = causal_conv1D(x, filters=16, length=3, stride=2)
	x = tf.math.add(x, x_skip3)

	# Block 6
	x_skip4 = causal_conv1D(x, filters=16, length=3, stride=2)
	x = tf.math.add(x, x_skip4)

	# Block 7
	#x_forward = causal_conv1D(x, filters=16, length=3, stride=2)
	x = causal_conv1D(x, filters=16, length=3, stride=2)

	# Block 8
	x_skip5 = causal_conv1D(x, filters=16, length=3, stride=2)
	
	# Output
	x = tf.concat([x_skip3, x_skip4, x, x_skip5], axis=-1)
	x = tf.compat.v1.layers.Conv1D(filters=classes, kernel_size=1)(x)
	x = tf.compat.v1.layers.AveragePooling1D(pool_size=[x.shape[1]], strides=1)(x)
	out = tf.compat.v1.layers.Flatten()(x)
	#out = tf.nn.softmax(x)

	return out
