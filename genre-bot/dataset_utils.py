import ast
import os
import os.path
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.svm
import sklearn.utils
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder,
                                   MultiLabelBinarizer, StandardScaler)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

AUDIO_DIR = "./data/fma/data/fma_small/"
SPECT_DIR = "./data/fma_small_spect/"  # TODO: Write method to save data


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def audio_to_spectrogram(audio_dir, track_id, spect_dir):

    filename = get_audio_path(audio_dir, track_id)
    y, sr = librosa.load(filename, sr=None, mono=True, duration=30)

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.axis('off')
    fig, ax = plt.subplots()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax)
    fig.savefig(os.path.join(spect_dir, '{:06d}'.format(track_id) + '.png'),  # TODO: remove white boarder
                format='png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=900)


def load_tracks(filepath):

    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    try:
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            'category', categories=SUBSETS, ordered=True)
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            pd.CategoricalDtype(categories=SUBSETS, ordered=True))

    COLUMNS = [('track', 'genre_top'), ('track', 'license'),
               ('album', 'type'), ('album', 'information'),
               ('artist', 'bio')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks


def testing(tracks, genres):

    # Select Small Subset
    small = tracks[tracks['set', 'subset'] <= 'small']
    if small.shape == (8000, 52):
        print("Small Subset Selected Successfully")
    else:
        print("Error: Something wrong with the dataset (wrong shape)")

    print(small)
    print("aaaa")
    print(small["track", "tags"])
    print(small["track", "tags"])

    # Select Genres
    print('{} top-level genres'.format(len(genres['top_level'].unique())))
    print(genres.loc[genres['top_level'].unique()].sort_values('#tracks', ascending=False))

    print(genres.sort_values('#tracks').head(10))

    # Load Audio
    filename = get_audio_path(AUDIO_DIR, 2)

    print('File: {}'.format(filename))

    x, sr = librosa.load(filename, sr=None, mono=True, duration=30)
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

    start, end = 7, 17
    # Librosa Stuff
    librosa.display.waveshow(x, sr=sr, alpha=0.5)
    plt.vlines([start, end], -1, 1)

    start = len(x) // 2
    plt.figure()
    plt.plot(x[start:start+2000])
    plt.ylim((-1, 1))
    plt.show()

    # stuff 2
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.db_to_amplitude(mel)

    librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.show()

    # stuff 3
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.show()

    y, sr = librosa.load(filename, sr=None, mono=True, duration=30)
    librosa.feature.melspectrogram(y=y, sr=sr)

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                       fmax=8000)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def generate_and_check_data(tracks, genres):

    # Select Small Subset
    small = tracks[tracks['set', 'subset'] <= 'small']
    if small.shape == (8000, 52):
        print("Small Subset Selected Successfully")
    else:
        print("Error: Something wrong with the dataset (wrong shape)")

    small_indices = small.index.to_list()
    print(f"Small Index: {small_indices}, {len(small_indices)}")

    # Select Genres
    print('{} top-level genres'.format(len(genres['top_level'].unique())))
    print(genres.loc[genres['top_level'].unique()].sort_values('#tracks', ascending=False))

    print(genres.sort_values('#tracks').head(10))

    # Load Audio
    filename = get_audio_path(AUDIO_DIR, 2)

    print('File: {}'.format(filename))

    x, sr = librosa.load(filename, sr=None, mono=True, duration=30)
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

    start, end = 7, 17
    # Librosa Stuff
    librosa.display.waveshow(x, sr=sr, alpha=0.5)
    plt.vlines([start, end], -1, 1)

    start = len(x) // 2
    plt.figure()
    plt.plot(x[start:start+2000])
    plt.ylim((-1, 1))
    plt.show()

    # stuff 2
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.db_to_amplitude(mel)

    librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.show()

    # stuff 3
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.show()


def main():

    # Load Metadata
    tracks = load_tracks('data/fma/data/fma_metadata/tracks.csv')
    print("Track Metadata Loaded Successfully")
    genres = pd.read_csv('data/fma/data/fma_metadata/genres.csv', index_col=0)
    print("Genre Metadata Loaded Successfully")

    # Select Small Subset
    small = tracks[tracks['set', 'subset'] <= 'small']
    if small.shape == (8000, 52):
        print("Small Subset Selected Successfully")
    else:
        print("Error: Something wrong with the dataset (wrong shape)")

    # Select Training, Validation and Testing Subsets
    train = tracks[(tracks['set', 'split'] == 'training') & (tracks['set', 'subset'] <= 'small')]
    val = tracks[(tracks['set', 'split'] == 'validation') & (tracks['set', 'subset'] <= 'small')]
    test = tracks[(tracks['set', 'split'] == 'test') & (tracks['set', 'subset'] <= 'small')]

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")  # TODO: debug
    # print(f"Train: {train.index}, Val: {val.index}, Test: {test.index}")  # TODO: debug

    small_indices = small.index.to_list()
    #print(f"Small Index: {small_indices}, {len(small_indices)}")

    GENRES = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
    genres_list = []
    for i in small_indices:
        genre = str(tracks.loc[i]['track', 'genre_top'])
        if genre not in genres_list:
            genres_list.append(genre)
    print(f"All genres in Small: {genres_list} {GENRES == genres_list}")

    print("First 10 track ids and genres:")
    for i in small_indices[:3]:
        print(i, tracks.loc[i]['track', 'genre_top'], tracks.loc[i]['track', 'title'])
        #print(i, tracks['track', 'genre_top'].loc[i], tracks['track', 'title'].loc[i])
        audio_to_spectrogram(AUDIO_DIR, i, SPECT_DIR)

    # labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])  # TODO: finish
    #labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)


if __name__ == "__main__":
    main()
