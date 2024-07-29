""" By Lennard Rose 5112737"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import sklearn.metrics as metrics
import itertools

plt.rcParams.update({'font.size': 14})


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"model saved to {path}")


def hit_accuracy(y_true, y_pred, tolerance=10):
    """
    compares two list for matching 1s. +- a tolerance of indices
    :param tolerance: the number of indices around a true hit to still count as a hit
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :return: score
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Labels and predictions are not of matching size!")

    count = 0
    for i in range(len(y_true)):
        lower_tolerance = max(0, i - tolerance)
        upper_tolerance = min(i + tolerance, len(y_pred))
        if y_true[i] == 1 and 1 in y_pred[lower_tolerance: upper_tolerance]:
            count += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            count += 1

    return count / len(y_true)


def get_device(cuda):
    """
    get the device to train on, only if available!
    :param cuda: True if GPU, False if CPU
    :return:
    """
    if cuda and torch.cuda.is_available():
        # Clear cache if non-empty
        torch.cuda.empty_cache()
        # See which GPU has been allotted
        print(f"Using cuda device: {torch.cuda.get_device_name(torch.cuda.current_device())} for training")
        return "cuda"
    else:
        print("Using cpu for training")
        return "cpu"


def plot_training(epochs, learning_rate, hidden_dims, batch_size,
                  training_losses, validation_losses, training_accuracies, validation_accuracies):
    """
    Plot 4 Subplots consisting of the Training and Validation losses and accuracies
    :param epochs:  The learning rate used during the training
    :param learning_rate: The learning rate used during the training
    :param hidden_dims: the hidden dimensions used for training, in case of cnn the output channels
    :param batch_size:  The learning rate used during the training
    :param training_losses: The training losses to display as a list
    :param validation_losses:  The Validation losses to display as a list
    :param training_accuracies:  The training accuracies to display as a list
    :param validation_accuracies:  The Validation accuracies to display as a list
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)
    fig.suptitle(f'Epochs: {epochs} LR: {learning_rate} Hidden Layers: {hidden_dims} Batch Size: {batch_size}')

    # losses
    axs[0, 0].plot(range(epochs), training_losses)
    axs[0, 0].set_title("Training Losses")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[1, 0].plot(range(epochs), validation_losses)
    axs[1, 0].set_title("Validation Losses")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].set_xlabel("Epochs")

    # accuracies
    axs[0, 1].plot(range(epochs), training_accuracies)
    axs[0, 1].set_title("Training Accuracies")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].set_xlabel("Epochs")
    axs[1, 1].plot(range(epochs), validation_accuracies)
    axs[1, 1].set_title("Validation Accuracies")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].set_xlabel("Epochs")

    plt.show()


def plot_3d(df_to_plot):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    activity = df_to_plot[" Activity"].iloc[0]
    print(f"Activity: {activity}")

    # Plotting the first set of coordinates
    axs[0].scatter(df_to_plot[' AccelX_5'], df_to_plot[' AccelY_5'], df_to_plot[' AccelZ_5'])
    axs[0].set_title('Accelerometer')
    axs[0].set_xlabel(' AccelX_5')
    axs[0].set_ylabel(' AccelY_5')
    axs[0].set_zlabel(' AccelZ_5')

    # Plotting the second set of coordinates
    axs[1].scatter(df_to_plot[' GyroX_5'], df_to_plot[' GyroY_5'], df_to_plot[' GyroZ_5'])
    axs[1].set_title('Gyroscope')
    axs[1].set_xlabel(' GyroX_5')
    axs[1].set_ylabel(' GyroY_5')
    axs[1].set_zlabel(' GyroZ_5')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_dataframes(dfs_to_plot, from_sample=0, to_sample=-1, steps=False):
    # Create a figure with 6 subplots
    fig, axs = plt.subplots(6, 1, figsize=(12, 8))

    for df_to_plot in dfs_to_plot:
        activity = df_to_plot[" Activity"].iloc[0]
        if from_sample < 0:  # as defaultvalue
            from_sample = len(df_to_plot) + from_sample
        if to_sample < 0:  # as defaultvalue
            to_sample = len(df_to_plot) + to_sample
        if steps:
            start_index = np.where(df_to_plot["start"] == 1)[0]
            end_index = np.where(df_to_plot["end"] == 1)[0]
            start_index = [value for value in start_index if from_sample <= value <= to_sample]
            end_index = [value for value in end_index if from_sample <= value <= to_sample]
            for i in range(6):
                # uses different color if overlapping
                for index in start_index:
                    axs[i].axvline(x=index, color='red')
                for index in end_index:
                    axs[i].axvline(x=index, color='green')

        axs[0].plot(df_to_plot[' AccelX_5'][from_sample:to_sample], label=activity)
        axs[0].set_title(' AccelX_5')

        axs[1].plot(df_to_plot[' AccelY_5'][from_sample:to_sample], label=activity)
        axs[1].set_title(' AccelY_5')

        axs[2].plot(df_to_plot[' AccelZ_5'][from_sample:to_sample], label=activity)
        axs[2].set_title(' AccelZ_5')

        axs[3].plot(df_to_plot[' GyroX_5'][from_sample:to_sample], label=activity)
        axs[3].set_title(' GyroX_5')

        axs[4].plot(df_to_plot[' GyroY_5'][from_sample:to_sample], label=activity)
        axs[4].set_title(' GyroY_5')

        axs[5].plot(df_to_plot[' GyroZ_5'][from_sample:to_sample], label=activity)
        axs[5].set_title(' GyroZ_5')

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.legend()

    # Display the plot
    plt.show()


def plot_roc(y_pred, y_true):
    """
    plots the ROC
    """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def output_results(result, title):
    """
    Creates a submittable dataframe from the test results and saves it
    for not classified test results (usually the last x when only predicting the first label)
    zeros are appended ( which is the most probable class)
    :param result: the test results to create a dataframe from
    :param title: the title of the output file
    """
    test_length = 102091

    padding = pd.Series(np.zeros(test_length - len(result)))
    first = pd.Series(result)
    first = pd.concat([first, padding], ignore_index=True)
    first = first.astype(int)

    second = first.shift(1)
    second.iloc[0] = 0
    second = second.astype(int)

    result = pd.concat([first, second], axis=1)
    result.columns = ["start", "end"]
    result.index.name = "index"
    result.to_csv(title)
    print(f"successfully saved results to {title}")


def plot_confusion_matrix(cm):
    """
    plots a confusion matrix in blue
    :param cm: the confusion matrix of sklearn
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
