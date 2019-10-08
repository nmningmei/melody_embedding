import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pathlib import Path

# GENERATE MEL SPECTROGRAMS FROM WAVEFORMS

def gen_spectrogram(data, sampling_rate, name, export_folder):

    plt.interactive(False)

    fig = plt.figure(figsize=[6, 6])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y = data, sr = sampling_rate)

    librosa.display.specshow(librosa.power_to_db(S, ref = np.max))

    filename = export_folder + name + '.png'

    plt.savefig(filename, dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

    fig.clf()

    plt.close(fig)

    plt.close('all')

    del data, sampling_rate, name, export_folder, fig, ax, S


# DEFINE MAIN VARIABLES & DIRECTORIES

sampling_rate = 22000

augmented_dir = 'data/augmented/'

spectrograms_dir = 'data/spectrograms/'


if __name__ == '__main__':

    # LOAD AUGMENTED MELODIES *.WAV FILES

    for filename in Path(augmented_dir).glob('*.wav'):

        filename = str(filename)

        wav_name = filename.split('/')[-1].split('.')[0]

        data, sampling_rate = librosa.load(filename, sr = sampling_rate)

        # GENERATE SPECTROGRAM FOR EACH FILE INSIDE AUGMENTED DIR

        gen_spectrogram(data = data,
                        sampling_rate = sampling_rate,
                        name = str(wav_name),
                        export_folder = spectrograms_dir,
                       )
