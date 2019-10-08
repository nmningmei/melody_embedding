import librosa
from pathlib import Path
import numpy as np


# DEFINE MAIN VARIABLES & DIRECTORIES

sampling_rate = 22000

augmented_dir = 'data/augmented/'

same_length_dir = 'data/same_length/'


if __name__ == '__main__':

    # MATCH AUGMENTED MELODIES LENGTH

    for filename in Path(augmented_dir).glob('*.wav'):

        filename = str(filename)

        wav_name = filename.split('/')[-1].split('.')[0]

        data, sampling_rate = librosa.load(filename, sr = sampling_rate)
    
        data_pad = np.concatenate([data, np.zeros(200000-data.shape[0])])

        np.save(f'{same_length_dir}{wav_name}.npy', data_pad)


