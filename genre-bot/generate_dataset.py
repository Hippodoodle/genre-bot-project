import ast
import os
import os.path
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIO_DIR = "./data/fma/data/fma_small"

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


def audio_to_spectrogram(audio_dir: str | Path, track_id: int, spect_dir: str | Path, dpi: int = 100):

    filename = get_audio_path(audio_dir, track_id)
    try:
        y, sr = librosa.load(filename, sr=None, mono=True, duration=30)
    except:
        return

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    fig, ax = plt.subplots()
    ax.axis('off')
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax)
    fig.savefig(os.path.join(spect_dir, '{:06d}'.format(track_id) + '.png'),  # TODO: remove white border
                format='png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=dpi)
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


def generate_dataset(genres: list[str], fresh: bool = True, pair_experiment: bool = False, dpi: int = 100):
    """Generates spectrogram dataset, stored by genre, from the small fma dataset

    Parameters
    ----------
    genres : list[str]
        List of genre string labels being used
    fresh : bool, optional
        Fresh load of all the data if set to True, by default True
    pair_experiment : bool, optional
        Generate datasets for binary choice experiments if set to True, by default False
    dpi : int, optional
        DPI setting for spectrogram generation, by default 100
    """

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

    # Generate genre pairs for experiments
    genre_sets: list[tuple[str, str]] | list[list[str]] = []
    if pair_experiment:
        for g1 in genres:
            for g2 in genres:
                if g1 < g2 and (g1, g2) not in genre_sets:
                    genre_sets.append((g1, g2))
                elif g1 > g2 and (g2, g1) not in genre_sets:
                    genre_sets.append((g2, g1))
    else:
        genre_sets = [genres]

    global SPECT_DIR
    if pair_experiment:
        SPECT_DIR = f"./data/binary_fma_small_spectrograms_dpi{dpi}"
    else:
        SPECT_DIR = f"./data/multiclass_{len(genre_sets[0])}_fma_small_spectrograms_dpi{dpi}"

    # Make directories if they don't exist
    if not os.path.exists(SPECT_DIR):
        os.makedirs(SPECT_DIR)

    for set in genre_sets:
        if len(genre_sets) > 1:
            experiment_dir = os.path.join(SPECT_DIR, set[0] + '_' + set[1])
        else:
            experiment_dir = SPECT_DIR

        train_dir = os.path.join(experiment_dir, 'training')
        validation_dir = os.path.join(experiment_dir, 'validation')
        test_dir = os.path.join(experiment_dir, 'test')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for g in set:
            if not os.path.exists(os.path.join(train_dir, g)):
                os.makedirs(os.path.join(train_dir, g))
            if not os.path.exists(os.path.join(validation_dir, g)):
                os.makedirs(os.path.join(validation_dir, g))
            if not os.path.exists(os.path.join(test_dir, g)):
                os.makedirs(os.path.join(test_dir, g))

    # Iterate through experiment genre sets
    for genre_set in genre_sets:

        # Load all tracks in the index list
        for id in small_indices:
            track_split = str(tracks.loc[id]['set', 'split'])

            if len(genre_sets) != 1:
                experiment_dir = os.path.join(SPECT_DIR, genre_set[0] + '_' + genre_set[1])
            else:
                experiment_dir = SPECT_DIR

            # Get track genre and path
            track_genre = get_track_genre(id, tracks)
            track_path = os.path.join(experiment_dir, track_split, track_genre)

            # Save a track if we want to save it and it has the desired genre
            if (not os.path.exists(get_spect_path(track_path, id)) or fresh) and track_genre in genre_set:
                audio_to_spectrogram(AUDIO_DIR, id, track_path)

        print(f"Generated data for {genre_set} set")


def main():

    start = time.time()

    # GENRES = ['Pop', 'Rock', 'Instrumental']
    GENRES = ['Hip-Hop', 'Pop', 'Folk', 'Rock', 'Instrumental']
    # GENRES = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']

    DPI = 100

    generate_dataset(genres=GENRES, fresh=True, pair_experiment=False, dpi=DPI)

    end = time.time()
    print(f'Data generation took {end - start} seconds')


if __name__ == "__main__":
    main()
