import os
import sys
from argparse import ArgumentParser

import librosa
import numpy as np
from scipy.io import wavfile


def change_extension(filename, new_extension):
    """
    Change the extension of the given file

    Parameters
    ----------
    filename : str
        The filename whose extension needs to be changed
    new_extension : str
        The new extension

    Return
    ------
    newfilename : str
        The new filename
    """
    dirname = os.path.dirname(filename)
    fname = os.path.basename(filename)
    basename = fname.split(".")[0]
    return os.path.join(dirname, "{}.{}".format(basename, new_extension))


def wav2mono(signal):
    """Converts a wav signal to mono. If the signal is already mono, it is not modified.

    Parameters
    ----------
    signal: ndarray
        The wav sound signal

    Return:
    -------
    signal: ndarray
        A mono wav signal
    """
    signal = np.squeeze(signal)
    if len(signal.shape) <= 1:
        return signal
    axis = 0 if signal.shape[0] == 2 else 1
    return np.mean(signal, axis=axis)


def extract_wave_files(path):
    """
    List the .wave file in the directory corresponding to `path`

    Parameters
    ----------
    path : str
        The path to a directory

    Return
    ------
    wave_files : list of str
        The wave files in `path`
    """
    walked = list(os.walk(path))
    all_files = list()
    for r, _, f in walked:
        all_files.extend([os.path.join(r, fi) for fi in f])
    return [f for f in all_files if f.endswith(".wav")]


def main(argv):
    parser = ArgumentParser(description='Converts a bunch of .wav file in .mfcc files.')
    parser.add_argument('-c', '--n_coef', dest="n_coef", type=int, default=13,
                        help='The number of coefficients to extract per input signal.')
    parser.add_argument('-f', '--files', dest='files', nargs='+', default=[],
                        help='The input .wav sound file to convert.')
    parser.add_argument('-p', "--path", dest="path", default=None,
                        help="Path in which the .wav files are located and must be converted "
                             "(ignored if some files are passed with the -f flag).")
    params = parser.parse_args(args=argv)

    files = params.files if len(params.files) > 0 or params.path is None else extract_wave_files(params.path)

    print("{} file(s) to be converted...".format(len(files)))

    for filename in np.unique(files):
        try:
            print("Converting '{}'...".format(filename))
            print(filename)
            sampling_rate, signal = wavfile.read(filename)

            signal = wav2mono(signal)  # to handle stereophonic signals

            mfcc = librosa.feature.mfcc(y=np.float32(signal), sr=sampling_rate, n_mfcc=params.n_coef)
            output_file = change_extension(filename, "mfcc")
            with open(output_file, "w+") as out:
                # out.write("{} {}".format(mfcc.shape[0], mfcc.shape[1]) + os.linesep)
                # print("{} {}".format(mfcc.shape[0], mfcc.shape[1]))
                title = ''
                for j in range(mfcc.shape[1]):
                    if j == 0:
                        title = title + '{},'.format(mfcc.shape[0])
                    elif j == 1:
                        title = title + '{},'.format(mfcc.shape[1])
                    else:
                        title = title + ','
                title = title[:-1] + '\n'
                out.write(title)

                for i in range(mfcc.shape[0]):
                    line = ''
                    for j in range(mfcc.shape[1]):
                        # out.write("{}".format(mfcc[i, j]))
                        line = line + '{},'.format(mfcc[i, j])
                    # out.write(os.linesep)
                    line = line[:-1] + '\n'
                    out.write(line)
        except IOError as e:
            print("Couldn't convert file '{}': {}".format(filename, e.strerror))


if __name__ == "__main__":
    """Dependencies: librosa, numpy, scipy
    Installation with anaconda (windows):
        - conda install scipy numpy
        - conda install -c conda-forge librosa=0.4.3

    Installation with pip (does not work on windows):
        - pip install scipy numpy librosa
    """
    main(sys.argv[1:])
