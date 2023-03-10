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
from ray.tune import CLIReporter, ExperimentAnalysis
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, random_split


DATA_DIR = "./data/multiclass_5_fma_small_spectrograms_dpi100"
RESULT_DIR = "./results/"
CLASSES = 5


class Net(nn.Module):
    def __init__(self, l1: int = 1024, l2: int = 512, classes: int = CLASSES):
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

    validationset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    return trainset, validationset, testset


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, validationset, testset = load_data(data_dir)

    #train_subset, validation_subset = random_split(trainset, [len(trainset) - len(testset), len(testset)])

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    validationloader = DataLoader(
        validationset,
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
    trainset, validationset, testset = load_data(data_dir)

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

    DATA_DIR = "./data/fma_small_spect_dpi100"

    data_dir = os.path.abspath(DATA_DIR)

    local_dir = os.path.join(RESULT_DIR, f"tuning_multiclass_{len(genres)}_genres", os.path.basename(data_dir))

    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 11)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 11)),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([1, 2, 4, 8, 16])
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([1, 2, 4, 8, 16])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"],
        max_report_frequency=60,
        print_intermediate_tables=True)

    result: ExperimentAnalysis = tune.run(
        partial(train_fma, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=local_dir)

    best_trial: Trial = result.get_best_trial("loss", "min", "last")  # type: ignore
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

    best_checkpoint_dir: str | Path = best_trial.checkpoint.dir_or_data  # type: ignore
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def experiment_run(g1: str, g2: str, num_samples: int = 1, max_num_epochs: int = 10, gpus_per_trial: int = 1, checkpoint: str | Path | None = None) -> str:

    DATA_DIR = "./data/fma_small_spect_dpi100_binary_choice"

    data_dir = None

    for filename in os.listdir(DATA_DIR):
        if g1.lower() in filename.lower() and g2.lower() in filename.lower():
            data_dir = os.path.abspath(os.path.join(DATA_DIR, filename))

    if data_dir is None:
        raise

    local_dir = os.path.join(RESULT_DIR, f"binary_{len(genres)}_genres", os.path.basename(data_dir))

    config = {
        "l1": 128,
        "l2": 512,
        "lr": 0.001,
        "batch_size": 1
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"],
        max_report_frequency=600,
        print_intermediate_tables=False)

    result: ExperimentAnalysis = tune.run(
        partial(train_fma, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=local_dir,
        verbose=0)

    best_trial: Trial = result.get_best_trial("loss", "min", "last")  # type: ignore
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

    best_checkpoint_dir: str | Path = best_trial.checkpoint.dir_or_data  # type: ignore
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device, data_dir)
    print("Best trial test set accuracy: {}".format(test_acc))

    # Output run data as csv string
    output_str = f"{g1},{g2},{test_acc}"

    return output_str


def simple_run(num_samples: int = 1, max_num_epochs: int = 10, gpus_per_trial: int = 1, checkpoint: str | Path | None = None):

    DATA_DIR = "./data/multiclass_5_fma_small_spectrograms_dpi100"

    data_dir = os.path.abspath(DATA_DIR)

    local_dir = os.path.join(RESULT_DIR, f"multiclass_{len(genres)}_genres", os.path.basename(data_dir))

    config = {
        "l1": 256,
        "l2": 64,
        "lr": 0.0018890629799798993,
        "batch_size": 4
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"],
        max_report_frequency=20,
        print_intermediate_tables=False)

    result: ExperimentAnalysis = tune.run(
        partial(train_fma, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=local_dir)

    best_trial: Trial = result.get_best_trial("loss", "min", "last")  # type: ignore
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

    best_checkpoint_dir: str | Path = best_trial.checkpoint.dir_or_data  # type: ignore
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def main():

    TUNE = False
    EXPERIMENT = False
    global CLASSES
    global genres

    MODEL_DIR = "C:/Users/thoma/ray_results/train_fma_2023-01-16_19-17-07"
    MODEL_DIR = None

    genres = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
    genres = ['Hip-Hop', 'Pop', 'Folk', 'Rock', 'Instrumental']  # 5 way binary choice experiment
    #genres = ['Pop', 'Rock', 'Instrumental']  # 3 way binary choice experiment

    if TUNE:
        CLASSES = 8
        genres = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
        tune_run(num_samples=100, max_num_epochs=20, gpus_per_trial=1)

    elif EXPERIMENT:
        CLASSES = 2
        experiment_csv = ""
        genre_pairs = []

        for g1 in genres:
            for g2 in genres:
                if g1 < g2 and not ([g1, g2] in genre_pairs or [g2, g1] in genre_pairs):
                    genre_pairs.append([g1, g2])

        for pair in genre_pairs:
            g1 = pair[0]
            g2 = pair[1]
            print(f"Current experiment: {g1}, {g2}")
            experiment_csv += experiment_run(g1, g2, max_num_epochs=10) + "\n"
            print("Output:\n" + experiment_csv)

        print("Final output:\n" + experiment_csv)
        f = open(os.path.join(RESULT_DIR, "experiment_output.csv"), "w")
        f.write(experiment_csv)
        f.close()

    else:
        CLASSES = 5
        genres = ['Hip-Hop', 'Pop', 'Folk', 'Rock', 'Instrumental']
        simple_run(max_num_epochs=10)


if __name__ == "__main__":
    main()
