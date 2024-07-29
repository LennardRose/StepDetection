""" By Lennard Rose 5112737"""

from torch import optim, nn
from torch.optim.lr_scheduler import ExponentialLR
from utils import *
from sklearn.metrics import accuracy_score
from tqdm.notebook import trange


def train_model(model, criterion, training_dataloader, test_dataloader, learning_rate, device, epochs, verbose, path_to_save, patience_epochs=10, evaluationfunc=accuracy_score):
    """
    Executes the training for a given model, initializes the data, Trains the model and validates the training,
    records and outputs the progress
    :param evaluationfunc:The evaluation function, default is accuracy
    :param model: The ANN Model to train
    :param training_dataloader: dataloader containing the training data
    :param learning_rate: learning rate for the adam optimizer
    :param test_dataloader: dataloader containing the validation data
    :param device: The device to execute the training/validation on (cpu/gpu)
    :param epochs: number of epochs to train the model
    :param verbose: Set True to output every  epochs loss/ accuracy
    :return: Training and Validation losses and accuracies
    """

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.8)

    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # early stopping mechanism
    loss_max = 100000
    patience = patience_epochs
    previous_decrease = False
    prev_loss = 0

    for epoch in trange(epochs, desc="Epochs"):

        ###################################### training #####################################################
        epoch_loss = 0
        prediction = []
        y_true = []

        model.train()
        for batch, labels in training_dataloader:

            torch.cuda.empty_cache()
            batch = batch.to(device)
            if model.activity:
                labels = labels.long().to(device) # crossentrophyloss excpects this shit
            else:
                labels = labels.float().to(device) #

            optimizer.zero_grad()

            output = model(batch)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            labels = labels.tolist()
            if model.activity:
                predictions = output.argmax(dim=1)
                predictions = predictions.tolist()
            else: #step detection
                predictions = output.tolist()
                if len(output.shape) >= 2:
                    predictions = [round(item) for window in predictions for item in window] # make the outputs 1 or 0
                    labels = [item for window in labels for item in window]
                else:
                    predictions = [round(item) for item in predictions] # make the outputs 1 or 0


            prediction += predictions
            y_true += labels
            epoch_loss += loss.item()

        # adjust learning rate, first after 10% of the iterations
        if epoch > epochs * 0.1:
            scheduler.step()

        # Loss and accuracy of current epoch
        avg_loss = epoch_loss / len(training_dataloader)
        accuracy = evaluationfunc(y_true, prediction) * 100
        training_losses.append(avg_loss)
        training_accuracies.append(accuracy)



        ###################################### validation #####################################################
        epoch_loss = 0
        prediction = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for batch, labels in test_dataloader:
                torch.cuda.empty_cache()
                batch = batch.to(device)

                if model.activity:
                    labels = labels.long().to(device) # crossentrophyloss excpects this shit
                else:
                    labels = labels.float().to(device) #

                output = model(batch)

                loss = criterion(output, labels)

                labels = labels.tolist()
                if model.activity:
                    predictions = output.argmax(dim=1)
                    predictions = predictions.tolist()
                else:  # step detection
                    predictions = output.tolist()
                    if len(output.shape) >= 2:
                        predictions = [round(item) for window in predictions for item in
                                       window]  # make the outputs 1 or 0
                        labels = [item for window in labels for item in window]
                    else:
                        predictions = [round(item) for item in predictions]  # make the outputs 1 or 0

                prediction += predictions
                y_true += labels
                epoch_loss += loss.item()

        # Loss and accuracy of current epoch
        val_loss = epoch_loss / len(test_dataloader)
        val_accuracy = evaluationfunc(y_true, prediction) * 100
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)

        # early stopping mechanism
        if prev_loss == round(val_loss, 3): # same result
            patience -= 1
            previous_decrease = True
            if patience == 0: # run out of patience
                print("Cancel - No progress")
                break
        elif prev_loss != val_loss and previous_decrease: # reset patience
            patience = 5
        prev_loss = round(val_loss, 3) # set for next

        if val_loss < loss_max:
            loss_max = val_loss
            save_model(model, path=path_to_save)
            print("saved model")

        # output
        if verbose:
            print(f"Epoch {epoch + 1} / {epochs}")
            print(f"Training Loss: {round(avg_loss, 3)} - Accuracy: {round(accuracy, 2)}%")
            print(f"Validation Loss: {round(val_loss, 3)} - Accuracy: {round(val_accuracy, 2)}%")

    return model, training_losses, training_accuracies, validation_losses, validation_accuracies


TEST_DATA_SIZE = 102091


def predict_steps(model, dataloader, device):
    """Method to make predictions with a given model and test samples.
    ONLY WORKS FOR BATCH/step SIZE 1
    Parameters
    ----------
    model : pytorch.model
        Trained Pytorch model.
    test_indices : list[int]
        List with some inidices for which the model should predict labels.
    """
    output = []
    sigmoid = nn.Sigmoid()

    model.eval()
    for X, y in dataloader:
        with torch.no_grad():
            X = X.to(device)
            y_hat = model(X)
            y_hat = y_hat.squeeze()
            y_hat = sigmoid(y_hat)
            y_hat = round(y_hat.item())
        output.append(y_hat)

    return output


def predict_activities(model, dataloader):
    """Method to make predictions with a given model and test samples.
    ONLY WORKS FOR BATCH/step SIZE 1
    Parameters
    ----------
    """
    output = []

    model.eval()
    for X, y in dataloader:
        with torch.no_grad():
            output.append(int(torch.argmax(model(X), 1).item()))

    padding = [output[-1]] * (TEST_DATA_SIZE - len(output)) # add missing samples, just assume the last one
    output = output + padding
    return pd.Series(output)


def predict_steps_average(model, dataloader, step_size=50):
    """
    Predictions based on averaging the results of a sliding window
    ONLY WORKS FOR BATCH SIZE 1
    ONLY WORKS if windowsize/stepsize % 2 = 0
    """
    sigmoid = nn.Sigmoid()
    result = np.full(TEST_DATA_SIZE, np.NaN)

    model.eval()
    for i, (X, y) in enumerate(dataloader):
        no_samples = len(X[0])
        with torch.no_grad():
            y_hat = model(X)

        output = y_hat.squeeze()
        output = sigmoid(output)
        output = output.cpu() #double check

        nan_before = i * step_size
        nan_after = TEST_DATA_SIZE - no_samples - nan_before
        before_padding = np.full(nan_before, np.NaN)  # nans in front of predictions
        after_padding = np.full(nan_after, np.NaN)  # nans after predictions
        resultcolumn = np.concatenate((before_padding, output.cpu().numpy(), after_padding), axis=0)

        if len(resultcolumn) != TEST_DATA_SIZE:
            raise ValueError("oops")

        result = np.nanmean(np.array([result, resultcolumn]), axis=0)

    result = np.rint(result)
    result = np.nan_to_num(result)

    return result
