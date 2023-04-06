import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from small_multiclass_train import Net, load_data, test_accuracy


@torch.no_grad()
def get_f1_score(net, data_dir: str | Path, device: str = "cpu", multiclass: bool = False):
    """calculate the f1 score of the model on the given data

    Parameters
    ----------
    net : Net
        model to be evaluated
    data_dir : str | Path
        path to data directory
    device : str, optional
        device to use, by default "cpu"

    Returns
    -------
    float
        f1 score
    """

    _, _, testset = load_data(data_dir)

    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    # Initialise lists for f1_score
    labels_list = []
    predicted_list = []

    # Iterate over all test data and label pairs
    for inputs, labels in testloader:
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)

        outputs = net(inputs)

        predicted: torch.Tensor
        _, predicted = torch.max(outputs.data, 1)

        actual_value = labels.tolist()[0]
        predicted_value = predicted.tolist()[0]

        labels_list.append(actual_value)
        predicted_list.append(predicted_value)

    if multiclass:
        score = (f1_score(labels_list, predicted_list, zero_division=0, average='micro'),  # type: ignore
                 f1_score(labels_list, predicted_list, zero_division=0, average='macro'))  # type: ignore
    else:
        score = f1_score(labels_list, predicted_list, zero_division=0)  # type: ignore

    return score


@torch.no_grad()
def generate_confusion_matrix(net, data_dir: str | Path, device: str = "cpu") -> pd.DataFrame:
    """Generate confusion matrix for given model and data inputs

    Parameters
    ----------
    net : Net
        model to be evaluated
    data_dir : str | Path
        path to data directory
    device : str, optional
        device to use, by default "cpu"

    Returns
    -------
    pd.DataFrame
        confusion matrix, as a DataFrame
    """

    _, _, testset = load_data(data_dir)

    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    # Initialise counters for confusion matrix
    confusion_matrix: np.ndarray = np.zeros((8, 8))

    # Iterate over all test data and label pairs
    for inputs, labels in testloader:
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)

        outputs = net(inputs)

        predicted: torch.Tensor
        _, predicted = torch.max(outputs.data, 1)

        actual_value = labels.tolist()[0]
        predicted_value = predicted.tolist()[0]

        """             actual
        [     1   2   3   4   5   6   7   8
          1 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          2 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          3 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          4 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          5 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          6 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          7 [ _ , _ , _ , _ , _ , _ , _ , _ ],
          8 [ _ , _ , _ , _ , _ , _ , _ , _ ]
        ]
        """

        confusion_matrix[predicted_value, actual_value] += 1

    true_labels = ["", "", "", "", "", "", "", ""]

    for class_label, index in testset.class_to_idx.items():
        true_labels[index] = class_label

    dataframe = pd.DataFrame(confusion_matrix, columns=true_labels, index=true_labels)

    return dataframe


def evaluate():

    CLASSES = 8
    DATA_DIR = f"C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/data/multiclass_{CLASSES}_fma_small_spectrograms_dpi100"
    RESULTS_DIR = f"C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/results/small_{CLASSES}_genres/"

    # Get data directory
    data_dir = DATA_DIR

    # Manually choose dir to evaluate
    checkpoint_dir = "C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/results/small_8_genres/multiclass_8_fma_small_spectrograms_dpi100/train_fma_2023-04-02_17-02-45/train_fma_ce888_00000_0_2023-04-02_17-02-45"

    # Get latest experiment directory if none manually chosen
    if checkpoint_dir is None:
        experiment_dir = os.path.join(RESULTS_DIR, os.path.basename(DATA_DIR))
        experiment_dir = max([os.path.join(experiment_dir, checkpoint_file) for checkpoint_file in os.listdir(experiment_dir)], key=os.path.getctime)

        # Get checkpoint directory
        checkpoint_dir = os.path.join(experiment_dir, [x for x in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, x))][0])

    best_f1_macro_score = 0
    best_checkpoint_path: str = ""

    # Iterate over checkpoints
    for i in range(30):

        # Get latest checkpoint file
        latest_checkpoint_path = os.path.join(checkpoint_dir,
                                              [x for x in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, x))][i],
                                              "checkpoint")

        # Initialise model and optimiser
        model = Net(CLASSES)

        # Load state dicts
        model_state, _ = torch.load(latest_checkpoint_path)
        model.load_state_dict(model_state)

        model.eval()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        model.to(device)

        _, f1_score_macro = get_f1_score(model, data_dir, device, multiclass=True)  # type: ignore

        if f1_score_macro > best_f1_macro_score:
            best_f1_macro_score = f1_score_macro
            best_checkpoint_path = latest_checkpoint_path

        del model

    # Initialise model and optimiser
    model = Net(CLASSES)

    # Load state dicts
    model_state, _ = torch.load(best_checkpoint_path)
    model.load_state_dict(model_state)

    model.eval()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    print(f"Best checkpoint is {os.path.basename(os.path.split(best_checkpoint_path)[0])}")

    test_acc = test_accuracy(model, data_dir, device)
    print(f"Accuracy for {CLASSES} genre multiclass experiment: {test_acc}")

    f1_score_micro, f1_score_macro = get_f1_score(model, data_dir, device, multiclass=True)  # type: ignore
    print(f"F1 score for {CLASSES} genre multiclass experiment: micro {f1_score_micro}, macro {f1_score_macro}")

    confusion_matrix = generate_confusion_matrix(model, data_dir, device)
    print(confusion_matrix)


def main():

    evaluate()


if __name__ == "__main__":
    main()
