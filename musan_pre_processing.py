import os

from pydub import AudioSegment, silence, effects


silent_file_counter = 0

def process(file, folder, category):
	global silent_file_counter

	# Load audio
	audio = AudioSegment.from_wav("musan/{}/{}/{}".format(category, folder, file))

	# Remove silence
	non_silent = silence.detect_nonsilent(audio, min_silence_len=50, silence_thresh=audio.dBFS-16)
	non_silent_audio = sum(audio[start:stop] for start,stop in non_silent)

	# Equalize loudness
	try:
		segments = non_silent_audio[::250]
		normalized_audio = sum(effects.normalize(segment) for segment in segments)

		output_file = "musan_pre_processed/{}/{}".format(category, file)
		normalized_audio.export(out_f = output_file, format = "wav")
	except:
		# Audio only contains silence
		silent_file_counter += 1


# Music
folders = os.listdir("musan/music")
folders.remove('README')

for folder in folders:
	print("Processing: ", folder)
	files = os.listdir("musan/music/{}".format(folder))
	files.remove('LICENSE')
	files.remove('ANNOTATIONS')

	if 'README' in files:
		files.remove('README')

	for file in files:
		#print(file)
		process(file, folder, "music")

# Noise
folders = os.listdir("musan/noise")
folders.remove('README')

for folder in folders:
	print("Processing: ", folder)
	files = os.listdir("musan/noise/{}".format(folder))
	files.remove('LICENSE')

	if 'ANNOTATIONS' in files:
		files.remove('ANNOTATIONS')

	if 'README' in files:
 		files.remove('README')
	
	for file in files:
		process(file, folder, "noise")

# Speech
folders = os.listdir("musan/speech")
folders.remove('README')

for folder in folders:
	print("Processing: ", folder)
	files = os.listdir("musan/speech/{}".format(folder))
	files.remove('LICENSE')

	if 'ANNOTATIONS' in files:
		files.remove('ANNOTATIONS')

	if 'README' in files:
 		files.remove('README')
	
	for file in files:
		process(file, folder, "speech")

print(silent_file_counter)