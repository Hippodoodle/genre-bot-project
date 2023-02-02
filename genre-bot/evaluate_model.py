import json
import os
from pathlib import Path

import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from train_model import Net, load_data, test_accuracy


@torch.no_grad()
def get_f1_score(net, data_dir: str | Path, device: str = "cpu"):
    trainset, testset = load_data(data_dir)

    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    score = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    list1 = []
    list2 = []
    for inputs, labels in testloader:
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        outputs = net(inputs)
        predicted: torch.Tensor
        _, predicted = torch.max(outputs.data, 1)
        list1.append(labels.tolist()[0])
        list2.append(predicted.tolist()[0])
        actual_value = labels.tolist()[0]
        predicted_value = predicted.tolist()[0]

    score = f1_score(labels.cpu().data, predicted.cpu(), zero_division=0)  # type: ignore
    score = f1_score(list1, list2, zero_division=0)  # type: ignore

    return score


def main():

    DATA_DIR = "./data/fma_small_spect_dpi100_binary_choice"
    RESULTS_DIR = "./results/experiment_5_genres/"
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


if __name__ == "__main__":
    main()
