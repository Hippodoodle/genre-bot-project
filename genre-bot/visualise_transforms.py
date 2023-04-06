import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


CLASSES = 8  # Can be changed to 3, 5 or 8 to limit classification over the music genre set


def get_mean_and_std(dataset: ImageFolder):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    mean = torch.mean(data[0], dim=(0, 2, 3))
    std = torch.std(data[0], dim=(0, 2, 3))
    return mean, std


def main():

    data_dir = f"./data/multiclass_{CLASSES}_fma_small_spectrograms_dpi100"

    #root=os.path.join(data_dir, "training")

    # Original dataset
    original = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=transforms.ToTensor())

    # Resized dataset with bilinear interpolation
    resized_bilinear = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    resized_bilinear = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=resized_bilinear)

    # Resized dataset with nearest neighbour interpolation
    nearest_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    resized_nearest_1 = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=nearest_transform)

    # Normalized dataset with colour channel means and stds of (0.5, 0.5, 0.5)
    normalize_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    resized_and_normalized = torchvision.datasets.ImageFolder(root=data_dir, transform=normalize_transform)

    # Standardized dataset with calculated mean and std of the whole dataset
    resized_nearest = torchvision.datasets.ImageFolder(root=data_dir, transform=nearest_transform)
    mean, std = get_mean_and_std(resized_nearest)

    standardize_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    resized_and_standardized = torchvision.datasets.ImageFolder(root=data_dir, transform=standardize_transform)

    # Check new mean and std
    print(get_mean_and_std(resized_nearest))
    print(get_mean_and_std(resized_and_standardized))
    print(get_mean_and_std(resized_and_normalized))

    # Create comparison figures

    # Interpolation comparison
    fig, ax = plt.subplots(ncols=2, squeeze=False)
    ax[0][0].imshow(transforms.ToPILImage()(resized_bilinear[0][0]))
    ax[0][0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Bilinear")
    ax[0][1].imshow(transforms.ToPILImage()(resized_nearest_1[0][0]))
    ax[0][1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Nearest Neighbour")
    fig.savefig("spectrogram_interpolation_comparison.pdf", dpi='figure', format='pdf', bbox_inches="tight")
    plt.close()

    # Resize comparison
    fig, ax = plt.subplots(ncols=2, squeeze=False)
    ax[0][0].imshow(transforms.ToPILImage()(original[0][0]))
    ax[0][0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="496 x 369 px")
    ax[0][1].imshow(transforms.ToPILImage()(resized_nearest_1[0][0]))
    ax[0][1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="224p x 224 px")
    fig.savefig("spectrogram_resize_comparison.pdf", dpi='figure', format='pdf', bbox_inches="tight")
    plt.close()

    # Normalize comparison
    image_range = 800  # run again to see prettier specs
    comparison_images = torch.stack((*[resized_nearest[i][0] for i in range(image_range, image_range + 5)], *[resized_and_normalized[i][0] for i in range(image_range, image_range + 5)], *[resized_and_standardized[i][0] for i in range(image_range, image_range + 5)]))
    plt.imshow(transforms.ToPILImage()(torchvision.utils.make_grid(comparison_images, nrow=5)))
    plt.axis('off')
    plt.savefig("spectrogram_normalize_comparison.pdf", dpi='figure', format='pdf', bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
