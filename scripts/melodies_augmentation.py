import librosa
import matplotlib.pyplot as plt
import librosa.display
import nlpaug.augmenter.audio as naa
import soundfile as sf
import nlpaug.flow as naf
from pathlib import Path
import numpy as np

# PLOT WAVEFORMS BEFORE & AFTER AUGMENTATION

def plot_aug_results(data, augmented_data, sampling_rate):

    plt.figure()

    librosa.display.waveplot(augmented_data, sr = sampling_rate, alpha = 0.5)

    librosa.display.waveplot(data, sr = sampling_rate, color='r', alpha = 0.25)

    plt.tight_layout()

    plt.show()


# DATA AUGMENTATION FROM WAVEFORMS

def augment_audio(data, sampling_rate, loudness_factor, speed_range, pitch_range, shift_max, noise_factor):

    """

    OPTIONS:

    LOUDNESS AUGMENTER: naa.LoudnessAug()
    MASK AUGMENTER: naa.MaskAug()
    SPEED AUGMENTER: naa.SpeedAug()
    SHIFT AUGMENTER: naa.ShiftAug()
    NOISE AUGMENTER: naa.NoiseAug()
    CROP AUGMENTER: naa.CropAug()
    PITCH AUGMENTER_ naa.PitchAug()

    """

    flow = naf.Sequential([naa.LoudnessAug(loudness_factor),
                           naa.SpeedAug(speed_range),
                           naa.PitchAug(sampling_rate = sampling_rate, pitch_range = pitch_range),
                           naa.ShiftAug(sampling_rate = sampling_rate, shift_max = shift_max),
                           naa.NoiseAug(noise_factor),
                           ])

    augmented_audio = flow.augment(data)

    return augmented_audio


# DEFINE MAIN VARIABLES & DIRECTORIES

resampling_rate = 22000

n_trials = 10

exp_melodies_dir = 'data/experimental_melodies/'

augmented_dir = 'data/augmented/'


if __name__ == '__main__':

    # LOAD ORIGINAL WAV FILES

    for filename in Path(exp_melodies_dir).glob('*.wav'):

        filename = str(filename)

        wav_name = filename.split('/')[-1].split('.')[0]

        data, sampling_rate = librosa.load(filename, sr = resampling_rate, mono = True, offset = 4)

        # PERFORM AUGMENTATION OVER EXPERIMENTAL MELODIES

        for trial in range(n_trials):

            augmented_audio = augment_audio(data,
                                            sampling_rate,
                                            loudness_factor = (0.8, 1.2),
                                            speed_range = (0.8, 1.2),
                                            pitch_range = (-2, 2),
                                            shift_max = 0.3,
                                            noise_factor = np.random.rand() / 10,
                                            )

            sf.write(f'{augmented_dir}{str(trial)}_{wav_name}.wav', augmented_audio, sampling_rate)

            # PLOT WAVEFORMS DIFFERENCES

            # plot_aug_results(data, augmented_audio, sampling_rate)