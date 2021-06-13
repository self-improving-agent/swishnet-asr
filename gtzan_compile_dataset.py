import sys
import numpy as np

if __name__ == "__main__":
	np.random.seed(42)
	size = lambda data, frac: int(np.ceil(len(data) * frac))

	# Load data
	music_samples = np.load("gtzan_pre_processed/music_two_sec.npy", allow_pickle=True)
	speech_samples = np.load("gtzan_pre_processed/speech_two_sec.npy", allow_pickle=True)

	# Shuffle data
	music_samples = music_samples[np.random.permutation(len(music_samples))]
	speech_samples = speech_samples[np.random.permutation(len(speech_samples))]

	# Extract datasets
	finetune_x = np.concatenate((music_samples[:size(music_samples, 0.25)], 
								 speech_samples[:size(speech_samples, 0.25)]))

	finetune_y = np.concatenate((np.zeros((size(music_samples, 0.25))), 
								 np.ones((size(speech_samples, 0.25)))))

	evaluate_x = np.concatenate((music_samples[size(music_samples, 0.25):], 
								 speech_samples[size(speech_samples, 0.25):]))

	evaluate_y = np.concatenate((np.zeros((len(music_samples)-size(music_samples, 0.25))), 
								 np.ones((len(speech_samples)-size(speech_samples, 0.25)))))

	# Shuffle each dataset and labels
	finetune_shuffle = np.random.permutation(len(finetune_x))
	finetune_x = finetune_x[finetune_shuffle]
	finetune_y = finetune_y[finetune_shuffle]

	evaluate_shuffle = np.random.permutation(len(evaluate_x))
	evaluate_x = evaluate_x[evaluate_shuffle]
	evaluate_y = evaluate_y[evaluate_shuffle]

	# Save results
	np.save("processed_datasets/gtzan_finetune_x.npy", finetune_x)
	np.save("processed_datasets/gtzan_finetune_y.npy", finetune_y)

	np.save("processed_datasets/gtzan_evaluate_x.npy", evaluate_x)
	np.save("processed_datasets/gtzan_evaluate_y.npy", evaluate_y)