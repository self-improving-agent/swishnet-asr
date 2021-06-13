import os

from pydub import AudioSegment, silence, effects
from logmmse import logmmse_from_file, logmmse

def process(file, category):
	# Apply logMMSE in place
	logmmse_from_file("gtzan/{}_wav/{}".format(category, file), "gtzan/{}_wav/{}".format(category, file))

	# Load audio
	audio = AudioSegment.from_wav("gtzan/{}_wav/{}".format(category, file))

	# Remove silence
	non_silent = silence.detect_nonsilent(audio, min_silence_len=50, silence_thresh=audio.dBFS-16)
	non_silent_audio = sum(audio[start:stop] for start,stop in non_silent)

	# Equalize loudness
	segments = non_silent_audio[::250]
	normalized_audio = sum(effects.normalize(segment) for segment in segments)

	output_file = "gtzan_pre_processed/{}/{}".format(category, file)
	normalized_audio.export(out_f = output_file, format = "wav")


music_files = os.listdir("gtzan/music_wav")

for file in music_files:
	process(file, "music")

speech_files = os.listdir("gtzan/speech_wav")

for file in speech_files:
	process(file, "speech")