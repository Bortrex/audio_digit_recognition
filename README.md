# Prediction of audio digits - DeepLearning

Audio digit transform into [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) representation and  its prediction using ConvNets.

<img src="https://github.com/Bortrex/audio_digit_recognition/assets/24497590/ebc2bd45-9aa8-4827-9b25-1615fcc1b8f4" width="900" height="400">

## Usage

To transform the audio digits into MFCC representation and train the model, run the following command:

    python wav2mfcc.py -c 13 -p path/to/audio

Where `-c` is the number of MFCC coefficients and `-p` is the path to the dataset.

It automatically handles the stereo signal into mono by taking the mean value of the two channels. 

In addition, to compensate for the different lengths of the audio signals, the MFCCs are filled with `-9999999` value to the maximum length of the dataset.



## Author

– [@Bortrex](https://github.com/Bortrex)
