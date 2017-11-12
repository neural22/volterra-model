from model import VolterraModel
import wave
import os
from scipy.io import wavfile

__author__ = 'aloriga'

test_memory = 250


def init_test():
    model = VolterraModel(order=3, memory=5)


def train_with_fake_signal():
    training_signal = [1, 2, 3, 4, 2, 7, 4]
    desired_signal = [4, 7, 9, 4, 8, 3, 1]
    model = VolterraModel(order=3, memory=2)
    model.train_with_signals(training_signal, desired_signal)


def train_with_white_noise():
    sampling_input, input_data = wavfile.read(os.path.join('.', 'data', 'input_white_noise.wav'))
    sampling_output, output_data = wavfile.read(os.path.join('.', 'data', 'output_white_noise.wav'))
    min_length = min([len(input_data), len(output_data)])
    print("Signal length {}".format(min_length))
    input_data = input_data[:min_length]
    output_data = output_data[:min_length]
    model = VolterraModel(order=1, memory=test_memory)
    model.train_with_signals(input_data, output_data, epochs=1, learning_rate=1e-16)


def inference_with_chirp(model_path):
    model = VolterraModel(order=3, memory=test_memory)
    _, input_signal = wavfile.read(os.path.join('.', 'data', 'chirp.wav'))
    model.inference(input_signal, path_load=model_path)


train_with_white_noise()
# inference_with_chirp(os.path.join('.', 'runs', '1510487391', 'checkpoints'))
