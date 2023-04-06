import json
import os

import matplotlib.pyplot as plt


def create_plots():

    # Manually choose dir to evaluate
    checkpoint_dir = "C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/results/multiclass_8_genres/multiclass_8_fma_small_spectrograms_dpi100/train_fma_2023-03-31_19-46-13/train_fma_4f8bb_00000_0_2023-03-31_19-46-13"
    checkpoint_dir = "C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/results/multiclass_8_genres/multiclass_8_fma_small_spectrograms_dpi100/train_fma_2023-03-31_22-55-14/train_fma_b7849_00000_0_2023-03-31_22-55-14"
    checkpoint_dir = "C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/results/multiclass_8_genres/multiclass_8_fma_small_spectrograms_dpi100/train_fma_2023-04-01_00-21-26/train_fma_c2637_00000_0_2023-04-01_00-21-26"

    # {'l1': 32, 'l2': 64, 'lr': 0.008564357333175636, 'momentum': 0.2887587012235814, 'batch_size': 8}
    checkpoint_dir = "C:/Users/thoma/Workspace/Uni/Year-4-Individual-Project/genre-bot-project/genre-bot/results/multiclass_8_genres/multiclass_8_fma_small_spectrograms_dpi100/train_fma_2023-04-01_22-32-20/train_fma_af3ac_00000_0_2023-04-01_22-32-21"

    # Load results
    result = [json.loads(line) for line in open(os.path.join(checkpoint_dir, "result.json"), 'r', encoding='utf-8')]

    train_loss = []
    val_loss = []
    epochs = range(1, 21)

    for row in result:
        train_loss.append(row['train_loss'])
        val_loss.append(row['loss'])

    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.xlim((0, 20))
    plt.grid(visible=True, which='major')
    plt.ylabel('Loss')
    plt.ylim((0, 4))
    plt.legend()
    plt.savefig("train_val_loss_graph.pdf", dpi='figure', format='pdf', bbox_inches=None, pad_inches=0)


def main():

    create_plots()


if __name__ == "__main__":
    main()
