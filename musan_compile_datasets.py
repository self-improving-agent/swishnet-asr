import sys
import numpy as np

if __name__ == "__main__":
	data = sys.argv[1] # First command line arg: which data to process
	set = sys.argv[2] # Second command line arg: dataset to create
	num_classes = sys.argv[3] # Third command line arg: number of classes
	np.random.seed(42)
	size = lambda data, frac: int(np.ceil(len(data) * frac))

	# Load data
	music_samples = np.load("musan_pre_processed/music_{}.npy".format(data), allow_pickle=True)
	noise_samples = np.load("musan_pre_processed/noise_{}.npy".format(data), allow_pickle=True)
	speech_samples = np.load("musan_pre_processed/speech_{}.npy".format(data), allow_pickle=True)

	# Shuffle data
	music_samples = music_samples[np.random.permutation(len(music_samples))]
	noise_samples = noise_samples[np.random.permutation(len(noise_samples))]
	speech_samples = speech_samples[np.random.permutation(len(speech_samples))]

	# Extract datasets
	# Shuffle each dataset and labels
	# Save results
	# (Separated to not run out of memory)
	if set == "train":
		train_x = np.concatenate((music_samples[:size(music_samples, 0.65)], 
								  noise_samples[:size(noise_samples, 0.65)],
								  speech_samples[:size(speech_samples, 0.65)]))
		# 2 class
		if num_classes == '2':
			train_y = np.concatenate((np.zeros((size(music_samples, 0.65)+size(noise_samples, 0.65))),
		 							  np.ones((size(speech_samples, 0.65)))))
		# 3 class
		elif num_classes == '3':
			train_y = np.concatenate((np.zeros((size(music_samples, 0.65))),
									  np.ones((size(noise_samples, 0.65))),
		 							  np.ones((size(speech_samples, 0.65)))+1))

		train_shuffle = np.random.permutation(len(train_x))
		train_x = train_x[train_shuffle]
		train_y = train_y[train_shuffle]

		np.save("processed_datasets/{}_{}_train_x.npy".format(data, num_classes), train_x)
		np.save("processed_datasets/{}_{}_train_y.npy".format(data, num_classes), train_y)

	elif set == "valid":
		valid_x = np.concatenate((music_samples[size(music_samples, 0.65):size(music_samples, 0.75)], 
								  noise_samples[size(noise_samples, 0.65):size(noise_samples, 0.75)],
								  speech_samples[size(speech_samples, 0.65):size(speech_samples, 0.75)]))
		# 2 class
		if num_classes == '2':
			valid_y = np.concatenate((np.zeros((size(music_samples, 0.75)-size(music_samples, 0.65)
												+size(noise_samples, 0.75)-size(noise_samples, 0.65))),
									  np.ones((size(speech_samples, 0.75)-size(speech_samples, 0.65)))))
		# 3 class
		elif num_classes == '3':
			valid_y = np.concatenate((np.zeros((size(music_samples, 0.75)-size(music_samples, 0.65))),
									  np.ones((size(noise_samples, 0.75)-size(noise_samples, 0.65))),
									  np.ones((size(speech_samples, 0.75)-size(speech_samples, 0.65)))+1))

		valid_shuffle = np.random.permutation(len(valid_x))
		valid_x = valid_x[valid_shuffle]
		valid_y = valid_y[valid_shuffle]

		np.save("processed_datasets/{}_{}_valid_x.npy".format(data, num_classes), valid_x)
		np.save("processed_datasets/{}_{}_valid_y.npy".format(data, num_classes), valid_y)

	elif set == "test":

		test_x = np.concatenate((music_samples[size(music_samples, 0.75):], 
								 noise_samples[size(noise_samples, 0.75):],
								 speech_samples[size(speech_samples, 0.75):]))
		# 2 class
		if num_classes == '2':

			test_y = np.concatenate((np.zeros((len(music_samples)-size(music_samples, 0.75)
									 		   +len(noise_samples)-size(noise_samples, 0.75))),
									 np.ones((len(speech_samples)-size(speech_samples, 0.75)))))
		# 3 class
		elif num_classes == '3':
			test_y = np.concatenate((np.zeros((len(music_samples)-size(music_samples, 0.75))),
								 	 np.ones((len(noise_samples)-size(noise_samples, 0.75))),
									 np.ones((len(speech_samples)-size(speech_samples, 0.75)))+1))

		test_shuffle = np.random.permutation(len(test_x))
		test_x = test_x[test_shuffle]
		test_y = test_y[test_shuffle]

		np.save("processed_datasets/{}_{}_test_x.npy".format(data, num_classes), test_x)
		np.save("processed_datasets/{}_{}_test_y.npy".format(data, num_classes), test_y)