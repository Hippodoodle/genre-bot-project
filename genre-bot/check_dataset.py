import os
from generate_dataset import load_tracks
import pandas as pd


def binary_check(data_dir: str, track_data: pd.DataFrame):

    track_ids = track_data.index.to_list()

    for filename in os.listdir(data_dir):
        no_repeats_counting_list = []

        for (split, i) in [("test", 0), ("training", 0), ("test", 1), ("training", 1)]:
            for track in os.listdir(os.path.join(data_dir, filename, split, filename.split("_")[i])):

                # Check genre of songs
                track_id = int(track[:6].lstrip('0'))
                assert str(track_data.loc[track_id]['track', 'genre_top']) == filename.split("_")[i]

                # Check for no repeats
                track_id = int(track[:6].lstrip('0'))
                no_repeats_counting_list.append(track_id)
                assert no_repeats_counting_list.count(track_id) == 1

            # Check dataset sizes
            subset_size = len(os.listdir(os.path.join(data_dir, filename, split, filename.split("_")[i])))
            if subset_size not in [100, 900]:
                print(f"filename: {filename}, split: {split}, genre: {filename.split('_')[i]}, subset_size: {subset_size}")
            elif split == "training":
                assert subset_size == 900
            elif split == "test":
                assert subset_size == 100
            else:
                assert split in ["training", "test"]


def multiclass_check(data_dir: str, track_data: pd.DataFrame):

    track_ids = track_data.index.to_list()

    """
    DATA_DIR
        |   Train
                |   Genre1
                        |   Track1.png
                        |   Track2.png
                        |    ...
                |   Genre2
                |    ...
        |   Validation
                |   Genre1
                |   Genre2
                |    ...
        |   Test
                |   Genre1
                |   Genre2
                |    ...
    """

    no_repeats_counting_list = []
    for split in os.listdir(data_dir):
        for genre in os.listdir(os.path.join(data_dir, split)):
            for track in os.listdir(os.path.join(data_dir, split, genre)):

                # Check that track id from file name is in id list
                track_id = int(track[:6].lstrip('0'))
                assert track_id in track_ids

                # Check genre of songs
                track_id = int(track[:6].lstrip('0'))
                assert str(track_data.loc[track_id]['track', 'genre_top']) == genre

                # Check for no repeats
                track_id = int(track[:6].lstrip('0'))
                no_repeats_counting_list.append(track_id)
                assert no_repeats_counting_list.count(track_id) == 1

            # Check split sizes
            subset_size = len(os.listdir(os.path.join(data_dir, split, genre)))
            if subset_size not in [100, 800]:
                print(f"split: {split}, genre: {genre}, subset_size: {subset_size}")
            elif split == "training":
                assert subset_size == 800
            elif split == "validation":
                assert subset_size == 100
            elif split == "test":
                assert subset_size == 100
            assert split in ["training", "validation", "test"]


def main():

    track_data = load_tracks('data/fma/data/fma_metadata/tracks.csv')
    track_data = track_data[track_data['set', 'subset'] <= 'small']

    # Check shape of metadata
    assert track_data.shape == (8000, 52)

    """
    DATA_DIR = "./data/fma_small_spect_dpi100_binary_choice"
    binary_check(data_dir=DATA_DIR, track_data=track_data)
    """

    DATA_DIR = "./data/multiclass_8_fma_small_spectrograms_dpi100"
    multiclass_check(data_dir=DATA_DIR, track_data=track_data)

    print("Dataset checks passed.")


if __name__ == "__main__":
    main()
