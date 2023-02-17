import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from train_model import Net, load_data, test_accuracy


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

    _, testset = load_data(data_dir)

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
        predicted_list.append(predicted_value)  # TODO: figure out true labels of data

    if multiclass:
        score = (f1_score(labels_list, predicted_list, zero_division=0, average='micro'),  # type: ignore
                 f1_score(labels_list, predicted_list, zero_division=0, average='macro'))  # type: ignore
    else:
        score = f1_score(labels_list, predicted_list, zero_division=0)  # type: ignore

    return score


@torch.no_grad()
def generate_binary_confusion_matrix(net, data_dir: str | Path, device: str = "cpu") -> pd.DataFrame:
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

    _, testset = load_data(data_dir)

    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    # Initialise counters for confusion matrix
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    # Iterate over all test data and label pairs
    for inputs, labels in testloader:
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)

        outputs = net(inputs)

        predicted: torch.Tensor
        _, predicted = torch.max(outputs.data, 1)

        actual_value = labels.tolist()[0]
        predicted_value = predicted.tolist()[0]

        if actual_value == 1 and predicted_value == 1:
            true_positive += 1
        elif actual_value == 1 and predicted_value == 0:
            false_negative += 1
        elif actual_value == 0 and predicted_value == 1:
            false_positive += 1
        elif actual_value == 0 and predicted_value == 0:
            true_negative += 1

    true_labels = ["", ""]

    for class_label, index in testset.class_to_idx.items():
        true_labels[index] = class_label

    confusion_matrix = pd.DataFrame([[true_positive, false_positive], [false_negative, true_negative]], columns=true_labels, index=true_labels)

    return confusion_matrix


@torch.no_grad()
def generate_multiclass_confusion_matrix(net, data_dir: str | Path, device: str = "cpu") -> pd.DataFrame:
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

    _, testset = load_data(data_dir)

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


def binary_evaluation():

    DATA_DIR = "./data/fma_small_spect_dpi100_binary_choice"
    RESULTS_DIR = "./results/binary_5_genres/"
    CLASSES = 2

    for experiment_basename in os.listdir(RESULTS_DIR):

        # Get data directory
        data_dir = os.path.join(DATA_DIR, experiment_basename)

        # Get experiment directory
        experiment_dir = os.path.join(RESULTS_DIR, experiment_basename)
        experiment_dir = max([os.path.join(experiment_dir, checkpoint_file) for checkpoint_file in os.listdir(experiment_dir)], key=os.path.getctime)

        # Get checkpoint directory
        checkpoint_dir = os.path.join(experiment_dir, [x for x in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, x))][0])

        # Load checkpoint config
        with open(os.path.join(checkpoint_dir, "params.json")) as json_file:
            config = json.load(json_file)

        # Get latest checkpoint file
        latest_checkpoint_path = os.path.join(checkpoint_dir,
                                              max([x for x in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, x))]),
                                              "checkpoint")

        # Initialise nn and optimiser
        model = Net(config["l1"], config["l2"], CLASSES)
        optimiser = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

        model_state, optimizer_state = torch.load(latest_checkpoint_path)
        model.load_state_dict(model_state)
        optimiser.load_state_dict(optimizer_state)

        model.eval()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        model.to(device)

        test_acc = test_accuracy(model, device, data_dir)
        print(f"Accuracy for {experiment_basename} experiment: {test_acc}")

        f1_score = get_f1_score(model, data_dir, device)
        print(f"F1 score for {experiment_basename} experiment: {f1_score}")

        confusion_matrix = generate_binary_confusion_matrix(model, data_dir, device)
        print(confusion_matrix)


def multiclass_evaluation():

    DATA_DIR = "./data/fma_small_spect_dpi100"
    RESULTS_DIR = "./results/multiclass_8_genres/"
    CLASSES = 8

    # Get data directory
    data_dir = DATA_DIR

    # Get experiment directory
    experiment_dir = os.path.join(RESULTS_DIR, "fma_small_spect_dpi100")
    experiment_dir = max([os.path.join(experiment_dir, checkpoint_file) for checkpoint_file in os.listdir(experiment_dir)], key=os.path.getctime)

    # Get checkpoint directory
    checkpoint_dir = os.path.join(experiment_dir, [x for x in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, x))][0])

    # Load checkpoint config
    with open(os.path.join(checkpoint_dir, "params.json")) as json_file:
        config = json.load(json_file)

    # Get latest checkpoint file
    latest_checkpoint_path = os.path.join(checkpoint_dir,
                                          max([x for x in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, x))]),
                                          "checkpoint")

    # Initialise nn and optimiser
    model = Net(config["l1"], config["l2"], CLASSES)
    optimiser = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    model_state, optimizer_state = torch.load(latest_checkpoint_path)
    model.load_state_dict(model_state)
    optimiser.load_state_dict(optimizer_state)

    model.eval()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    test_acc = test_accuracy(model, device, data_dir)
    print(f"Accuracy for {CLASSES} genre multiclass experiment: {test_acc}")

    f1_score = get_f1_score(model, data_dir, device, multiclass=True)
    print(f"F1 score for {CLASSES} genre multiclass experiment: {f1_score}")

    confusion_matrix = generate_multiclass_confusion_matrix(model, data_dir, device)
    print(confusion_matrix)


def main():

    #binary_evaluation()

    multiclass_evaluation()


if __name__ == "__main__":
    main()
