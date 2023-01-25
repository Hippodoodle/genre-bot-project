import os
from dataset_utils import load_tracks


def main():

    DATA_DIR = "./data/fma_small_spect_dpi100_binary_choice"

    track_data = load_tracks('data/fma/data/fma_metadata/tracks.csv')
    track_data = track_data[track_data['set', 'subset'] <= 'small']

    track_ids = track_data.index.to_list()

    # Check shape of metadata
    assert track_data.shape == (8000, 52)

    # Check genre of songs
    for filename in os.listdir(DATA_DIR):
        for (split, i) in [("test", 0), ("training", 0), ("test", 1), ("training", 1)]:
            for track in os.listdir(os.path.join(DATA_DIR, filename, split, filename.split("_")[i])):
                track_id = int(track[:6].lstrip('0'))
                assert str(track_data.loc[track_id]['track', 'genre_top']) == filename.split("_")[i]

    # Check for no repeats
    for filename in os.listdir(DATA_DIR):
        counting_list = []
        for (split, i) in [("test", 0), ("training", 0), ("test", 1), ("training", 1)]:
            for track in os.listdir(os.path.join(DATA_DIR, filename, split, filename.split("_")[i])):
                track_id = int(track[:6].lstrip('0'))
                counting_list.append(track_id)
                assert counting_list.count(track_id) == 1

    # Check dataset sizes
    for filename in os.listdir(DATA_DIR):
        for (split, i) in [("test", 0), ("training", 0), ("test", 1), ("training", 1)]:
            subset_size = len(os.listdir(os.path.join(DATA_DIR, filename, split, filename.split("_")[i])))
            if subset_size not in [100, 900]:
                print(f"filename: {filename}, split: {split}, genre: {filename.split('_')[i]}, subset_size: {subset_size}")
            elif split == "training":
                assert subset_size == 900
            elif split == "test":
                assert subset_size == 100
            else:
                assert split in ["training", "test"]

    print("Dataset checks passed.")


if __name__ == "__main__":
    main()
