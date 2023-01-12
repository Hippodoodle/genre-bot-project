import ast
import os
import os.path
import time

import librosa
import librosa.display
import matplotlib
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

AUDIO_DIR = "./data/fma/data/fma_small"
SPECT_DIR = "./data/fma_small_spect_dpi100_binary_choice"

DPI = 100

GENRES = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']

matplotlib.use("Agg")


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def get_spect_path(spect_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    spect_dir = os.path.join(spect_dir, )
    return os.path.join(spect_dir, tid_str + '.png')


def get_track_genre(track_id: int, tracks: pd.DataFrame) -> str:
    return str(tracks.loc[track_id]['track', 'genre_top'])


def get_track_id_from_path(track_path):
    return str(track_path)[-10:-4].lstrip('0')


def track_spectrogram_exists(spect_dir, track_id):
    return os.path.exists(get_spect_path(spect_dir, track_id))


def audio_to_spectrogram(audio_dir, track_id, spect_dir):

    filename = get_audio_path(audio_dir, track_id)
    try:
        y, sr = librosa.load(filename, sr=None, mono=True, duration=30)
    except:
        return

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.axis('off')
    plt.gca().axis('off')
    plt.margins(0, 0)
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax)
    fig.savefig(os.path.join(spect_dir, '{:06d}'.format(track_id) + '.png'),  # TODO: remove white border
                format='png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=DPI)
    plt.close('all')


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


def generate_and_check_data(tracks: pd.DataFrame, genres: pd.DataFrame, subset_str='small', ):

    # Select Subset
    subset = tracks[tracks['set', 'subset'] <= subset_str]
    if subset.shape == (8000, 52):
        print(f"Success {subset_str} subset selected")
    else:
        print("Error: Something wrong with the dataset (wrong shape)")

    small_indices = small.index.to_list()


def generate_data(path: str, genres: list[str], indices: list[int], tracks: pd.DataFrame, fresh: bool):

    # Load all tracks in the index list
    for id in indices:
        track_split = str(tracks.loc[id]['set', 'split'])

        # Merge validation split into training split
        if track_split == 'validation':
            track_split = 'training'

        # Get track genre and path
        track_genre = get_track_genre(id, tracks)
        track_path = os.path.join(SPECT_DIR, track_split, track_genre)

        # Save a track if we want to save it and it has the desired genre
        if (not os.path.exists(get_spect_path(track_path, id)) or fresh) and track_genre in genres:
            print(id, get_track_genre(id, tracks), tracks.loc[id]['track', 'title'], tracks.loc[id]['set', 'split'])
            audio_to_spectrogram(AUDIO_DIR, id, track_path)


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


def get_track_genre(track_id: int, tracks: pd.DataFrame) -> str:
    return str(tracks.loc[track_id]['track', 'genre_top'])


def generate_and_check_data(tracks: pd.DataFrame, genres: pd.DataFrame, subset_str='small', ):

    # Select Subset
    subset = tracks[tracks['set', 'subset'] <= subset_str]
    if subset.shape == (8000, 52):
        print(f"Success {subset_str} subset selected")
    else:
        print("Error: Something wrong with the dataset (wrong shape)")

    small_indices = small.index.to_list()


def main():

    # Load Metadata
    tracks = load_tracks('data/fma/data/fma_metadata/tracks.csv')
    print("Track Metadata Loaded Successfully")

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

    small_indices = small.index.to_list()

    genres_list = []
    for id in small_indices:
        genre = str(tracks.loc[id]['track', 'genre_top'])
        if genre not in genres_list:
            genres_list.append(genre)
    print(f"All genres in Small: {genres_list} {GENRES == genres_list}")

    # Limit genres for initial proof of concept
    genres_list = ['Folk', 'Rock']

    # Make directories if they don't exist
    if not os.path.exists(SPECT_DIR):
        os.makedirs(SPECT_DIR)
    train_dir = os.path.join(SPECT_DIR, 'training')
    # validation_dir = os.path.join(SPECT_DIR, 'validation')
    test_dir = os.path.join(SPECT_DIR, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for g in genres_list:
        if not os.path.exists(os.path.join(train_dir, g)):
            os.makedirs(os.path.join(train_dir, g))
        if not os.path.exists(os.path.join(test_dir, g)):
            os.makedirs(os.path.join(test_dir, g))

    # Load all tracks in the index list
    fresh = True  # Fresh load of all the data if set to True
    for id in small_indices:
        track_split = str(tracks.loc[id]['set', 'split'])
        if track_split == 'validation':
            track_split = 'training'
        track_genre = get_track_genre(id, tracks)
        track_path = os.path.join(SPECT_DIR, track_split, track_genre)
        if (not os.path.exists(get_spect_path(track_path, id)) or fresh) and track_genre in genres_list:
            print(id, get_track_genre(id, tracks), tracks.loc[id]['track', 'title'], tracks.loc[id]['set', 'split'])
            audio_to_spectrogram(AUDIO_DIR, id, track_path)


if __name__ == "__main__":
    main()
