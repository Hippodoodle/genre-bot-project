import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


CLASSES = 3  # Can be changed to 3, 5 or 8 to limit classification over the music genre set


def main():

    data_dir = f"./data/multiclass_{CLASSES}_fma_small_spectrograms_dpi100"

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    normalize_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    original = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=transforms.ToTensor())

    resized = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=transform)

    resized_and_normalized = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "training"), transform=normalize_transform)

    comparison_images = torch.stack((*[resized[i][0] for i in range(5)], *[resized_and_normalized[i][0] for i in range(5)]))

    plt.imshow(transforms.ToPILImage()(torchvision.utils.make_grid(comparison_images, nrow=5)))
    plt.axis('off')
    plt.savefig("spectrogram_normalize_comparison.png", bbox_inches="tight", dpi=200)
    plt.close()

    fig, ax = plt.subplots(ncols=2, squeeze=False)
    ax[0][0].imshow(transforms.ToPILImage()(original[0][0]))
    ax[0][0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="496 x 369 px")
    ax[0][1].imshow(transforms.ToPILImage()(resized[0][0]))
    ax[0][1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="256p x 256 px")
    fig.savefig("spectrogram_resize_comparison.png", bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
