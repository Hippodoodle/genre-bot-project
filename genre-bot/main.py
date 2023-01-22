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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, random_split


DATA_DIR = "./data/fma_small_spect_dpi100"
RESULT_DIR = "./result/"


class Net(nn.Module):
    def __init__(self, l1: int = 1024, l2: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256*32*32, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, classes)
        )

    def forward(self, x):
        return self.network(x)


def load_data(data_dir: str | Path = DATA_DIR):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=transform)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    return trainset, testset


def train_fma(config, checkpoint_dir: str | Path | None = None, data_dir: str | Path = DATA_DIR):

    start = time.time()

    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.parallel.DataParallel(net)
    net.to(device)
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(f"Using {device} device")
    print(net)
    print(classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, validation_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    validationloader = DataLoader(
        validation_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print some statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(validationloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    end = time.time()
    print(f'Training took {end - start} seconds')


def test_accuracy(net, device: str = "cpu", data_dir: str | Path = DATA_DIR):
    trainset, testset = load_data(data_dir)

    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def tune_run(num_samples: int = 1, max_num_epochs: int = 10, gpus_per_trial: int = 1, checkpoint: str | Path | None =  None):

    data_dir = os.path.abspath(DATA_DIR)
    load_data(data_dir)

    """
    if num_samples == 1:
            config = {
                "l1": 1024,
                "l2": 512,
                "lr": 0.001,
                "batch_size": 1
            }
        if num_samples == 1:
            config = {
                "l1": 64,
                "l2": 64,
                "lr": 0.001,
                "batch_size": 2
            }
    """
    if num_samples == 1:
        config = {
            "l1": 128,
            "l2": 512,
            "lr": 0.001,
            "batch_size": 1
        }
    else:
        config = {
            # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 11)),
            # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 11)),
            # "lr": tune.loguniform(1e-4, 1e-1),
            # "batch_size": tune.choice([1, 2, 4, 8, 16])
            "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
            "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([1, 2, 4])
        }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"],
        max_report_frequency=20)

    result = tune.run(
        partial(train_fma, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial: Trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.parallel.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir: str | Path = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    TUNE = False
    global classes
    if TUNE:
        classes = 8
    else:
        classes = 8
        main()
