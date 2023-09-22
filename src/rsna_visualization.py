import matplotlib.pyplot as plt
import numpy as np

def plot_image_gallery(images, value_range=(0, 1), rows=2, cols=2):
    """ Plot a gallery of images. """
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(images):
                image = images[idx].numpy().transpose(1, 2, 0)
                image = np.clip(image, *value_range)
                axes[i, j].imshow(image)
                axes[i, j].axis('off')
    plt.show()

def plot_per_task_accuracy(logger):
    # Create a 3x2 grid for the subplots
    fig, axes = plt.subplots(5, 1, figsize=(5, 15))
    # Flatten axes to iterate through them
    axes = axes.flatten()
    # Iterate through the metrics and plot them
    for i, name in enumerate(["bowel", "extra", "kidney", "liver", "spleen"]):
        # Plot training accuracy
        axes[i].plot(logger.logged_metrics['train_' +name + '_accuracy'], label='Training ' + name)
        # Plot validation accuracy
        axes[i].plot(logger.logged_metrics['val_' + name + '_accuracy'], label='Validation ' + name)
        axes[i].set_title(name)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_loss(logger):
    plt.plot(logger.logged_metrics["train_loss_epoch"], label="loss")
    plt.plot(logger.logged_metrics["val_loss"], label="val loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def get_best_epoch(logger):
    # Find the best epoch based on validation loss
    best_epoch = np.argmin(logger.logged_metrics['val_loss'])
    best_loss = logger.logged_metrics['val_loss'][best_epoch]
    best_acc_bowel = logger.logged_metrics[f'val_bowel_accuracy'][best_epoch]
    best_acc_extra = logger.logged_metrics[f'val_extra_accuracy'][best_epoch]
    best_acc_liver = logger.logged_metrics[f'val_liver_accuracy'][best_epoch]
    best_acc_kidney = logger.logged_metrics[f'val_kidney_accuracy'][best_epoch]
    best_acc_spleen = logger.logged_metrics[f'val_spleen_accuracy'][best_epoch]

    # Calculate mean accuracy
    best_acc = np.mean([best_acc_bowel, best_acc_extra, best_acc_liver, best_acc_kidney, best_acc_spleen])

    print(f'>>>> BEST Loss  : {best_loss:.3f}\n>>>> BEST Acc   : {best_acc:.3f}\n>>>> BEST Epoch : {best_epoch}\n')
    print('ORGAN Acc:')
    print(f'  >>>> {"Bowel".ljust(15)} : {best_acc_bowel:.3f}')
    print(f'  >>>> {"Extravasation".ljust(15)} : {best_acc_extra:.3f}')
    print(f'  >>>> {"Liver".ljust(15)} : {best_acc_liver:.3f}')
    print(f'  >>>> {"Kidney".ljust(15)} : {best_acc_kidney:.3f}')
    print(f'  >>>> {"Spleen".ljust(15)} : {best_acc_spleen:.3f}')