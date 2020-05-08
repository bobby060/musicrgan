import os
import sys

"""
Usage:
python convert_audio.py <train dir in> <test dir in> <train dir out> <test dir out> <bitrate>

Works for files without special characters
Make sure to use relative paths to data
"""

"""
Command:
ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav

https://stackoverflow.com/questions/13358287/how-to-convert-any-mp3-file-to-wav-16khz-mono-16bit
https://janakiev.com/blog/python-shell-commands/

"""


def convert(input, output):
    for subdir, dirs, files in os.walk(input):
        for file in files:
            # print os.path.join(subdir, file)
            if file.endswith(".mp3"):
                filepath = subdir + os.sep + file
                print(filepath)
                new_file = file.replace(".mp3", ".wav")
                output_path = output + os.sep + new_file
                command_string = "ffmpeg -i " + filepath + " -acodec pcm_s16le -ac 1 -ar " + str(bitrate) + " " + output_path
                os.system(command_string)


argv = sys.argv
input_dir_test = argv[1]
input_dir_train = argv[2]
output_dir_test = argv[3]
output_dir_train = argv[4]
bitrate = argv[5]

convert(input_dir_train, output_dir_train)
convert(input_dir_test, output_dir_test)