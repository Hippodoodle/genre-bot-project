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
from torchvision.transforms import InterpolationMode

torch.manual_seed(3407)


class Net(nn.Module):
    def __init__(self, classes: int, l1: int = 1024, l2: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            # 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
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

    # Send to cuda device if one is available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.parallel.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])

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
        train_loss = 0.0
        train_steps = 0
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

            train_loss += loss.item()
            train_steps += 1

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

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total, train_loss=(train_loss / train_steps))


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


def binary_train(g1: str, g2: str, num_classes: int, data_dir: str, result_dir: str, max_num_epochs: int = 10, gpus_per_trial: int = 1):

    for filename in os.listdir(data_dir):
        if g1.lower() in filename.lower() and g2.lower() in filename.lower():
            data_dir = os.path.abspath(os.path.join(data_dir, filename))
            break

    local_dir = os.path.join(result_dir, f"binary_{len(genres)}_genres", os.path.basename(data_dir))

    config = {'l1': 32, 'l2': 64, 'lr': 0.008564357333175636, 'momentum': 0.2887587012235814, 'batch_size': 8}

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

    _: ExperimentAnalysis = tune.run(
        partial(train_fma, num_classes=num_classes, data_dir=data_dir, num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=local_dir,
        verbose=0)


def main():
    """Train a binary classifying model for each pair of music genre
    """

    global genres

    genres = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']

    DATA_DIR = "../data/binary_fma_small_spectrograms_dpi100"
    RESULT_DIR = "../results/"
    genre_pairs = []

    for g1 in genres:
        for g2 in genres:
            if g1 < g2 and not ([g1, g2] in genre_pairs or [g2, g1] in genre_pairs):
                genre_pairs.append([g1, g2])

    start = time.time()

    for g1, g2 in genre_pairs:
        print(f"Current experiment: {g1}, {g2}")
        binary_train(g1, g2, num_classes=2, data_dir=DATA_DIR, result_dir=RESULT_DIR, max_num_epochs=30)

    end = time.time()
    print(f'Training took {end - start} seconds')


if __name__ == "__main__":
    main()
