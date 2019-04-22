from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm


import config as cfg
import file_io as io
import utils_tf
import utils

from attacks import deep_fool_method as DFM
from cleverhans_wrapper import CleverHansModel


def _load_dataset(dataset):
    """Load input data and the associated metadata for a dataset.

    Args:
        dataset: Structure encapsulating dataset information.

    Returns:
        tuple: Tuple containing:

            x (np.ndarray): The input data of the dataset.
            df (pd.DataFrame): The metadata of the dataset.
    """
    import features

    # Load feature vectors and reshape to 4D tensor
    features_path = os.path.join(cfg.extraction_path, dataset.name + '.h5')
    x, n_chunks = utils.timeit(lambda: features.load_features(features_path),
                               'Loaded features of %s dataset' % dataset.name)
    x = np.expand_dims(x, -1)
    assert x.ndim == 4

    # Load metadata and duplicate entries based on number of chunks
    df = io.read_metadata(dataset.metadata_path)

    return x, df



def _preprocess_data(audio_path,audio_file_name):
    filelist = os.listdir(audio_path)
    #print('Audio file name:', audio_file_name)
    

    audio_file_path = os.path.join(audio_path,audio_file_name)
    data,q = _load_audio(audio_file_path)
    return data,q

def _get_label_from_audio(audio_path,audio_file_name,metadata_path):
    filelist = os.listdir(audio_path)
    df = pd.read_csv(metadata_path)
    for iter in range(df.shape[0]):
        temp = df.loc[iter]['fname']
        temp = temp[:-4]

        if(temp in audio_file_name):
            return df.loc[iter]['label']

def _convert_label_name_to_label(label_name):
    label=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
                
    return label.index(label_name)
def _convert_label_to_label_name(iter):
    label=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
    label_name = label[iter]
    return label_name

def  _load_audio(audio_file_path):
    data,fs = librosa.load(audio_file_path)
    data = librosa.resample(data,fs,32000)
    l = 128*512
    r = data.shape[0]%l
    q = np.ceil(data.shape[0]/l)
    if (r==0):
        return data,q
    else:
        return np.pad(data,(0,l-r),'constant',constant_values=(0)),q


def _compute_features(pcm,sample_rate):
    
    stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=512,
                                           fft_length=1024,pad_end=True)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0, 16000, 64
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
              num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
              tf.pow(spectrograms,2), linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
         linear_to_mel_weight_matrix.shape[-1:]))

#    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = 10*tf.log(mel_spectrograms + 1e-6)
    log_mel_spectrograms = tf.reshape(log_mel_spectrograms,shape=[-1,128,64,1])
 


    return log_mel_spectrograms

def _compute_librosa_features(pcm,sample_rate):
    pcm = librosa.resample(pcm,sample_rate,32000)
    D = librosa.stft(pcm,n_fft=1024,hop_length=512)
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64)
    S = np.dot(mel_fb,np.abs(D)**2).T
    return librosa.power_to_db(S,ref=np.max,top_db=None)

def _reshape_spec(feat,r_threshold=32):
    q = feat.shape[0] // 128
    r = feat.shape[0] % 128
    r_threshold = 32
    #print(q,r)

    if not q:
        split = [utils.pad_truncate(feat, 128, pad_value=np.min(feat))]
    else:
        off = r // 2 if r < r_threshold else 0
        split = np.split(feat[off:q * 128 + off], q)
        if r >= r_threshold:
            split.append(feat[-128:])
        return np.array(split)


