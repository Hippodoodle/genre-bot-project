import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            # 224 x 224 x 3
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1),  # 222 x 222 x 6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 111 x 111 x 6
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1),  # 109 x 109 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 54 x 54 x 16
            nn.Flatten(),
            nn.Linear(16 * 54 * 54, 120),
            nn.ReLU(),
            nn.Linear(120, 36),
            nn.ReLU(),
            nn.Linear(36, classes)
        )

    def forward(self, x):
        return self.network(x)


def load_data(data_dir: str | Path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=transform)

    validationset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    return trainset, validationset, testset


def train_fma(config, num_classes: int, data_dir: str | Path, num_epochs: int):

    net = Net(classes=num_classes)

    # Send to cuda device if one is available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.parallel.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    trainset, validationset, _ = load_data(data_dir)

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


def multiclass_train(num_classes: int, data_dir: str, result_dir: str, max_num_epochs: int = 10, gpus_per_trial: int = 1):

    data_dir = os.path.abspath(data_dir)

    local_dir = os.path.join(result_dir, f"multiclass_{num_classes}_genres", os.path.basename(data_dir))

    config = {
        "lr": 0.001,
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
        max_report_frequency=60,
        print_intermediate_tables=False)

    _: ExperimentAnalysis = tune.run(
        partial(train_fma, num_classes=num_classes, data_dir=data_dir, num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=local_dir)


def main():

    CLASSES = 3  # Can be changed to 3, 5 or 8 to limit classification over the music genre set
    DATA_DIR = f"./data/multiclass_{CLASSES}_fma_small_spectrograms_dpi100"
    RESULT_DIR = "./results/"

    start = time.time()

    multiclass_train(num_classes=CLASSES, data_dir=DATA_DIR, result_dir=RESULT_DIR, max_num_epochs=10)

    end = time.time()
    print(f'Training took {end - start} seconds')


if __name__ == "__main__":
    main()
