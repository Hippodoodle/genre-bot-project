import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from main import Net, test_accuracy
from ray import tune
from ray.tune import CLIReporter, ExperimentAnalysis
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, random_split


def main():

    DATA_DIR = "./data/fma_small_spect_dpi100_binary_choice"
    RESULTS_DIR = "./results/experiment_5_genres/"
    CLASSES = 2

    for experiment_dir in os.listdir(RESULTS_DIR):
        experiment_dir = os.path.join(RESULTS_DIR, experiment_dir)
        experiment_dir = max([os.path.join(experiment_dir, checkpoint_file) for checkpoint_file in os.listdir(experiment_dir)], key=os.path.getctime)

        checkpoint_dir = os.path.join(experiment_dir,
                                      [x for x in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, x))][0])

        # Load checkpoint config
        with open(os.path.join(checkpoint_dir, "params.json")) as json_file:
            config = json.load(json_file)

        # Get latest checkpoint file
        latest_checkpoint = os.path.join(checkpoint_dir,
                                         max([x for x in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, x))]),
                                         "checkpoint")

        # Initialise nn and optimiser
        model = Net(config["l1"], config["l2"], CLASSES)
        optimiser = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

        model.eval()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        model.to(device)

        """
        best_trial: Trial = result.get_best_trial("loss", "min", "last")  # type: ignore
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        best_trained_model.to(device)

        best_checkpoint_dir: str | Path = best_trial.checkpoint.dir_or_data  # type: ignore
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))
        """


if __name__ == "__main__":
    main()
