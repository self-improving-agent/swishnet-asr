import os
import sys
import librosa
import numpy as np
import time

from pydub import AudioSegment
from python_speech_features import mfcc
from signal import signal, SIGINT


# Handler to save the progress upon interruption
def handler(signal_received, frame):
	np.save("musan_pre_processed/{}_half_sec.npy".format(category), half_sec_samples)
	np.save("musan_pre_processed/{}_sec.npy".format(category), sec_samples)
	np.save("musan_pre_processed/{}_two_sec.npy".format(category), two_sec_samples)
	sys.exit(0)

# Generate ranges
def get_ranges(signal_length, range_length):
	half_ranges = range(0, signal_length, range_length // 2)
	ranges = []

	for i in range(len(half_ranges)-2):
		ranges.append((half_ranges[i], half_ranges[i+2]))

	if len(half_ranges) > 1:
		ranges.append((half_ranges[-2], signal_length))
	else:
		ranges.append((half_ranges[0], signal_length))

	return ranges

# Get features in a particular set of ranges
def process_in_range(signal, ranges, frames):
	clip_features = []

	# Get features for segments
	for start, end in ranges:
		segment = np.array(signal[start:end].get_array_of_samples()).astype('float32')
		mfcc = librosa.feature.mfcc(segment, sr=sample_rate, n_fft=int(0.025*sample_rate), hop_length=int(0.015*sample_rate))
		clip_features.append(mfcc.T)

	# Resize for uniformity
	clip_features[-1] = np.resize(clip_features[-1], (frames, 20))

	return clip_features

# Main wrapper
def process_clip(signal):
	half_sec_ranges = get_ranges(len(signal), 500)
	half_sec_features = process_in_range(signal, half_sec_ranges, 34)

	sec_ranges = get_ranges(len(signal), 1000)
	sec_features = process_in_range(signal, sec_ranges, 67)

	two_sec_ranges = get_ranges(len(signal), 2000)
	two_sec_features = process_in_range(signal, two_sec_ranges, 134)

	return half_sec_features, sec_features, two_sec_features


if __name__ == "__main__":
	signal(SIGINT, handler)
	category = sys.argv[1] # First command line arg: category to process

	# Change to change starting point (for continuing)
	start = 1

	# Load incomplete files or initialise new ones
	if os.path.exists("musan_pre_processed/{}_half_sec.npy".format(category)):
		half_sec_samples = np.load("musan_pre_processed/{}_half_sec.npy".format(category), allow_pickle=True)
		sec_samples = np.load("musan_pre_processed/{}_sec.npy".format(category), allow_pickle=True)
		two_sec_samples = np.load("musan_pre_processed/{}_two_sec.npy".format(category), allow_pickle=True)
	else:
		half_sec_samples = np.zeros((1,34,20))
		sec_samples = np.zeros((1,67,20))
		two_sec_samples = np.zeros((1,134,20))

	files = os.listdir("musan_pre_processed/{}/".format(category))

	# Iterate over files
	for i in range(start-1, len(files)):
		print("Progress: {}/{}".format(i+1, len(files)))

		# Load file and get rate
		clip = AudioSegment.from_wav("musan_pre_processed/{}/{}".format(category,files[i]))
		sample_rate = clip.frame_rate

		# Get processed samples
		half_sec, sec, two_sec = process_clip(clip)

		# Add to list
		half_sec_samples = np.concatenate((half_sec_samples, half_sec))
		sec_samples = np.concatenate((sec_samples, sec))
		two_sec_samples = np.concatenate((two_sec_samples, two_sec))

	# Remove initial only zeros entry
	half_sec_samples = np.delete(half_sec_samples, 0, 0)
	sec_samples = np.delete(sec_samples, 0, 0)
	two_sec_samples = np.delete(two_sec_samples, 0, 0)

	# Save files
	np.save("musan_pre_processed/{}_half_sec.npy".format(category), half_sec_samples)
	np.save("musan_pre_processed/{}_sec.npy".format(category), sec_samples)
	np.save("musan_pre_processed/{}_two_sec.npy".format(category), two_sec_samples)
