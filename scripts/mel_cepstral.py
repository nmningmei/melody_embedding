import librosa.display
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler


# DEFINE MAIN VARIABLES & DIRECTORIES

sampling_rate = 22000

max_pad_len = 400

augmented_dir = 'data/augmented/'

mel_cepstral_dir = 'data/mel_cepstral/'


if __name__ == '__main__':

    # LOAD AUGMENTED MELODIES *.WAV FILES

    for filename in Path(augmented_dir).glob('*.wav'):

        filename = str(filename)

        wav_name = filename.split('/')[-1].split('.')[0]

        data, sampling_rate = librosa.load(filename, sr = sampling_rate)

        # GENERATE MEL FREQUENCY CEPSTRAL COEFFICIENTS FOR AUGMENTED MELODIES

        mfccs = librosa.feature.mfcc(data, sr = sampling_rate, n_mfcc = 100)

        # SCALE MEL CEPSTRAL COEFFICIENTS FOR AUGMENTED MELODIES

        scaler = RobustScaler()

        mfccs = scaler.fit_transform(mfccs)

        # librosa.display.specshow(mfccs, sr = sampling_rate, x_axis = 'time')

        pad_width = max_pad_len - mfccs.shape[1]

        mfccs_scaled = np.pad(mfccs, pad_width = ((0, 0), (0, pad_width)), mode='constant')

        np.save(f'{mel_cepstral_dir}{wav_name}.npy', mfccs_scaled)