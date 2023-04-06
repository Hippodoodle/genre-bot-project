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
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode


torch.manual_seed(3407)


class Net(nn.Module):
    def __init__(self, classes: int, l1: int = 1024, l2: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            # 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 112 x 112 x 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 56 x 56 128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 28 x 28 x 256
            nn.Flatten(),
            nn.Linear(28 * 28 * 256, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, classes)
        )

    def forward(self, x):
        return self.network(x)


def load_data(data_dir: str | Path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=transform)

    validationset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    return trainset, validationset, testset


def train_fma(config, num_classes: int, data_dir: str | Path, num_epochs: int):

    net = Net(classes=num_classes, l1=config["l1"], l2=config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])

    trainset, validationset, _ = load_data(data_dir)

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0)
    validationloader = DataLoader(
        validationset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
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
            if epoch > 4:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)


def test_accuracy(net, data_dir: str | Path, device: str = "cpu"):
    _, _, testset = load_data(data_dir)

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


def tune_run(num_classes: int, data_dir: str, result_dir: str, num_samples: int = 20, max_num_epochs: int = 10, gpus_per_trial: int = 1):

    data_dir = os.path.abspath(data_dir)

    local_dir = os.path.join(result_dir, f"tuning_multiclass_{num_classes}_genres", os.path.basename(data_dir))

    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 10)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(5, 10)),
        "lr": tune.loguniform(1e-4, 1e-2),
        "momentum": tune.uniform(0.0, 1.0),
        "batch_size": tune.choice([1, 2, 4, 8, 16])
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
        print_intermediate_tables=True)

    result: ExperimentAnalysis = tune.run(
        partial(train_fma, num_classes=num_classes, data_dir=data_dir, num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=local_dir,
        verbose=0)

    best_trial: Trial = result.get_best_trial("loss", "min", "all")  # type: ignore
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    best_trained_model = Net(classes=num_classes, l1=best_trial.config["l1"], l2=best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.parallel.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir: str | Path = best_trial.checkpoint.dir_or_data  # type: ignore
    model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, data_dir=data_dir, device=device)
    print("Best trial test set accuracy: {}".format(test_acc))


def main():

    CLASSES = 8  # Can be changed to 3, 5 or 8 to limit classification over the music genre set
    DATA_DIR = f"./data/multiclass_{CLASSES}_fma_small_spectrograms_dpi100"
    RESULT_DIR = "./results/"

    start = time.time()

    tune_run(num_classes=CLASSES, data_dir=DATA_DIR, result_dir=RESULT_DIR, num_samples=100, max_num_epochs=15, gpus_per_trial=1)

    end = time.time()
    print(f'Tuning took {end - start} seconds')


if __name__ == "__main__":
    main()
