""" By Lennard Rose 5112737"""

import glob
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader

pd.options.mode.chained_assignment = None  # default='warn' --> ignore, does not apply here


def load_dataframes(data_path="./data/**/*.csv",
                    label_path="./data/**/*.csv.stepMixed",
                    labels="keep",
                    steps="binary"):
    """
    loads a pandas dataframe from a given label and datafile, as well as aligning them.
    :param data_path: path to the data csv. Default is for local
    :param label_path: path to the label csv.stepMixed data. Default is local
    :param labels: Which labels to keep (keep = all, activity = " Activity", start/end = respective step
    :param steps: if steps schould be binary or continuous
    :return:
    """
    data_files = glob.glob(data_path)
    label_files = glob.glob(label_path)
    dataframes = []

    # align
    data_files.sort()
    label_files.sort()

    for datafile, labelfile in zip(data_files, label_files):
        # make sure label and data files are matching
        if datafile[-41:-1] != "Clipped" + labelfile[-44:-11]:
            raise ValueError("Files not matching")
        # load contents
        dataframe = pd.read_csv(datafile, index_col=None)
        labelframe = pd.read_csv(labelfile, index_col=None, header=None)
        # split labels to start and end
        starts = labelframe.iloc[:, 0]
        ends = labelframe.iloc[:, 1]
        if steps == "binary":
            # convert labels indizes to 0/1 columns
            dataframe["start"] = np.zeros(len(dataframe))
            dataframe["start"][starts] = 1
            dataframe["end"] = np.zeros(len(dataframe))
            dataframe["end"][ends] = 1
        else:  # continuous
            dataframe["start"] = make_steps_continuous(starts, len(dataframe), THRESHOLD_STD)
            dataframe["end"] = make_steps_continuous(ends, len(dataframe), THRESHOLD_STD)
        # append
        dataframes.append(dataframe)

    return_df = pd.concat(dataframes, ignore_index=True)

    # what labels to use for the label column
    if labels == "activity":
        return_df["label"] = return_df[" Activity"]
        return_df = return_df.drop(['start', 'end', " Activity"], axis=1)
    elif labels == "start":
        return_df["label"] = return_df["start"]
        return_df = return_df.drop(['start', 'end', " Activity"], axis=1)
    elif labels == "end":
        return_df["label"] = return_df["end"]
        return_df = return_df.drop(['start', 'end', " Activity"], axis=1)
    else:  # default keep
        pass  # keep all columns as is

    return return_df


# NOT IMPORTANT, CONTINUOUS LABELS NOT USED
# std if steps are made continuous, calculated in the exploration.
START_STD = 6
END_STD = 6
# No of samples around the stepindex that also count as hit
THRESHOLD_STD = 10


def make_steps_continuous(step_indices, dataframe_length, std=THRESHOLD_STD):
    """
    Convert binary steps to continuous values. steps stay at value 1, surroundings gauss distributed (small values > 0.4).
    :param step_indices: the indices of the steps valued 1
    :param dataframe_length: the length of the dataframe for the continuous step values
    :param std: the std determining the spread around the step indices
    :return: a pandas series with the continuous step values
    """
    # Create an array of indices
    indices = np.arange(dataframe_length)

    # Create an empty array to store the values
    values = np.zeros(dataframe_length)

    # Iterate over each mean (step)
    for step in step_indices:
        # Calculate the probabilities of the normal distribution
        probs = norm.pdf(indices, step, std)
        # Calculate the values based on the mean and probabilities
        values += std * probs

    # 1 at the step indices
    values[step_indices] = 1

    # Create the Pandas Series
    return pd.Series(values)


def sort_dataframe_by_length(df_list):
    """
    Sort DESCENDING by df length
    :param df_list: list of dataframes to sort
    :return: sortet list of dataframes
    """
    return sorted(df_list, key=lambda df: len(df), reverse=True)


def split_dataframe_by_activity(df):
    """
    splits dataframe by activity
    :param df: the dataframe to split
    :return: 4 dataframes each one with one activity
    """
    df_0 = df[df[" Activity"] == 0].reset_index(drop=True)
    df_1 = df[df[" Activity"] == 1].reset_index(drop=True)
    df_2 = df[df[" Activity"] == 2].reset_index(drop=True)
    df_3 = df[df[" Activity"] == 3].reset_index(drop=True)

    return df_0, df_1, df_2, df_3


def cut_df(df_to_cut, end):
    """
    cuts dataframe to a specific lenght
    :param df_to_cut: the dataframe
    :param end: the index to cut
    :return: the cutted dataframe
    """
    print(f"cut class of {len(df_to_cut)} samples down to {end} samples total")
    return df_to_cut[0:end]


def expand_df(df_to_expand, factor):
    """
    expands given dataframe by a factor, just appending the same data to the end
    :param df_to_expand: the dataframe to expand
    :param factor: the factor to multiply the dataframe with
    :return: the expanded dataframe
    """
    print(f"expand class of {len(df_to_expand)} samples by factor {factor}")
    return df_to_expand.append([df_to_expand] * (factor - 1))


def SMOTE_resample(df, label):
    """
    expands a dataframe using the SMOTE method (synthetic minority oversampling technique) which extends the minority
    class by sampling from its distribution. Better dont use this for time series.
    :param df: the dataframe to extend
    :param label: label to keep for your dataframe
    :return: the extended dataframe
    """
    # Separate the features and the label
    if label == "activity":
        label = " Activity"  # hacky

    y = pd.DataFrame({'label': df[label]})
    X = df.drop([' Activity', 'start', 'end'], axis=1)

    # Create an instance of SMOTE with the desired sampling strategy
    smote = SMOTE(sampling_strategy="auto")

    # Perform SMOTE oversampling
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)


def interpolate_resample(df_to_interpolate, factor):
    """
    interpolates values of a dataframe by adding interpolated values evently between samples
    :param df_to_interpolate: the dataframe to extend
    :param factor: the amount of interpolated samples between all real samples
    :return: the interpolated dataframe
    """
    print(f"Add interpolated samples to dataframe of {len(df_to_interpolate)} samples by factor {factor}")
    # Add n blank rows
    new_index = pd.RangeIndex(len(df_to_interpolate) * factor)
    new_df = pd.DataFrame(np.nan, index=new_index, columns=df_to_interpolate.columns)
    ids = np.arange(len(df_to_interpolate)) * factor
    new_df.loc[ids] = df_to_interpolate.values
    new_df = new_df.interpolate()
    return new_df


def get_balanced_dataset(data_path="./data/**/*.csv",
                         label_path="./data/**/*.csv.stepMixed",
                         method="expand",
                         labels="keep"):
    dataframe = load_dataframes(data_path=data_path, label_path=label_path, labels="keep")
    df_list = list(split_dataframe_by_activity(dataframe))
    df_list = sort_dataframe_by_length(df_list)
    return_df = []

    # Balance based on Method specified

    if method == "expand":
        for df in df_list:
            expand_factor = round(len(df_list[0]) / len(df))  # longest df divided by current df floor
            if df_list[0][" Activity"].iloc[0] != df[" Activity"].iloc[0]:  # check for same list
                return_df.append(expand_df(df, factor=expand_factor))
            else:  # the dataframe that others get expanded to
                return_df.append(df)

        return_df = pd.concat(return_df)

    if method == "cut":
        for df in df_list:
            shortest = len(df_list[-1])  # shortest df
            return_df.append(cut_df(df, end=shortest))
        return_df = pd.concat(return_df)

    if method == "SMOTE":
        return SMOTE_resample(dataframe, label=labels)

    if method == "interpolate":
        for df in df_list:
            expand_factor = round(len(df_list[0]) / len(df))  # longest df divided by current df floor
            if df_list[0][" Activity"].iloc[0] != df[" Activity"].iloc[0]:  # check for same list
                return_df.append(interpolate_resample(df, factor=expand_factor))
            else:  # the dataframe that others get expanded to
                return_df.append(df)
        return_df = pd.concat(return_df)

    # set label
    if labels == "activity":
        return_df["label"] = return_df[" Activity"]
        return_df = return_df.drop(['start', 'end', " Activity"], axis=1)
    elif labels == "start":
        return_df["label"] = return_df["start"]
        return_df = return_df.drop(['start', 'end', " Activity"], axis=1)
    elif labels == "end":
        return_df["label"] = return_df["end"]
        return_df = return_df.drop(['start', 'end', " Activity"], axis=1)
    else:  # default keep
        pass  # keep all columns as is

    return return_df


def get_training_dataloader(batch_size, scaler, labels):
    """
    get a simple dataloader with unaltered data, contrary to sliding window
    :param batch_size: the batchsize for the dataloader
    :param scaler: the scaler to scale the data with
    :param labels: the label to keep
    :return: dataloader training, dataloader validation, fitted scaler
    """
    data = load_dataframes(labels=labels)
    y = data["label"]
    x = data.drop(columns=["label"])

    split_index = int(len(data) * 0.9)

    x_train = x[0:split_index]
    x_val = x[split_index: -1]

    y_train = y[0:split_index]
    y_val = y[split_index: -1]

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_data = torch.utils.data.TensorDataset(torch.Tensor(x_train.to_numpy()),
                                                torch.LongTensor(y_train.to_numpy()))

    val_data = torch.utils.data.TensorDataset(torch.Tensor(x_val.to_numpy()),
                                              torch.LongTensor(y_val.to_numpy()))

    return DataLoader(train_data, batch_size=batch_size, shuffle=True), \
           DataLoader(val_data, batch_size=batch_size, shuffle=True), \
           scaler


def get_SlidingWindow_Dataloader(df,
                                 scaler,
                                 batch_size,
                                 ordered_df=False,
                                 train_test_split=0.9,
                                 window_size=100,
                                 step_size=50,
                                 train=True,
                                 shape_for="lstm",
                                 target_index="n_to_n"
                                 ):
    """
    creates a sliding window dataloader
    :param df: the dataframe to create sliding windows from
    :param scaler: the scaler to scale the data with
    :param batch_size: batch size for the dataloader
    :param ordered_df: True if the dataframe was balanced
    :param train_test_split: factor to split the dataset (0.9 = 90% train)
    :param window_size: the size of the sliding window
    :param step_size: number of indizes to step
    :param train: if the dataset is for training purposes
    :param shape_for: transfrom the dataset for the cause "lstm", "cnn" "resnet" possible
    :param target_index: "first", "last", "n_to_n" which index is the label
    :return: sliding window dataloader train, val, scaler if train, test, none, scaler if test
    """
    # generate slidingwindow dataset
    data, labels = create_sliding_window(df=df, window_size=window_size, step_size=step_size, train=train,
                                         shape_for=shape_for, target_index=target_index)

    # set train/val
    if train:
        # set index to split train/val
        split_index = int(len(data) * train_test_split)

        if ordered_df:
            # the df is balanced, therefore the data is ordered by their acitivities,
            # to prevent the validation set only containing one activity the indices have to be shuffled
            indices = np.arange(len(data))
            indices = np.random.permutation(indices)

            train_indices = indices[:split_index]
            val_indices = indices[split_index: -1]

            x_train = torch.index_select(input=data, dim=0, index=torch.Tensor(train_indices).type(torch.int32))
            x_val = torch.index_select(input=data, dim=0, index=torch.Tensor(val_indices).type(torch.int32))

            y_train = torch.index_select(input=labels, dim=0, index=torch.Tensor(train_indices).type(torch.int32))
            y_val = torch.index_select(input=labels, dim=0, index=torch.Tensor(val_indices).type(torch.int32))

        else:  # unordered df

            x_train = data[0:split_index]
            x_val = data[split_index: -1]

            y_train = labels[0:split_index]
            y_val = labels[split_index: -1]

        # free ram or colab colabses hehe
        data = None
        labels = None

        original_shape = x_train.shape
        x_train = x_train.reshape(-1, x_train.shape[-1])  # reshape to scale
        x_train = scaler.fit_transform(x_train).reshape(original_shape)  # scale and shape back

        original_shape = x_val.shape
        x_val = x_val.reshape(-1, x_val.shape[-1])  # reshape to scale
        x_val = scaler.transform(x_val).reshape(original_shape)  # scale and shape back

        # create datasets
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train),
                                                       torch.LongTensor(y_train.numpy()))

        val_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_val),
                                                     torch.LongTensor(y_val.numpy()))
        # free ram or colab colabses hehe
        x_train = None
        x_val = None

        # rerturn dataloaders
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
               DataLoader(val_dataset, batch_size=batch_size, shuffle=False), \
               scaler

    else:  # for test

        # scale
        original_shape = data.shape
        data = data.reshape(-1, data.shape[-1])  # reshape to scale
        data = scaler.transform(data).reshape(original_shape)  # scale and shape back

        # create datasets
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(data),
                                                      torch.LongTensor(labels.numpy()))

        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False), \
               None, \
               scaler


def create_sliding_window(df,
                          window_size=100,
                          step_size=50,
                          train=True,
                          target_index="n_to_n",
                          shape_for="lstm"):
    """creates a sliding window
    :param df: the dataframe to create sliding windows from
    :param window_size: the size of the sliding window
    :param step_size: number of indizes to step
    :param train: if the dataset is for training purposes
    :param shape_for: transfrom the dataset for the cause "lstm", "cnn" "resnet" possible
    :param target_index: "first", "last", "n_to_n" which index is the label
    :return: sliding window data, sliding window labels
    """
    windows = []
    labels = []

    for i in range(0, len(df) - window_size, step_size):

        end_index = i + window_size

        # append values to window, gets transposed later
        windows.append([df[' AccelX_5'].values[i: end_index],
                        df[' AccelY_5'].values[i: end_index],
                        df[' AccelZ_5'].values[i: end_index],
                        df[' GyroX_5'].values[i: end_index],
                        df[' GyroY_5'].values[i: end_index],
                        df[' GyroZ_5'].values[i: end_index]])

        if train:
            if target_index == "n_to_n":
                labels.append(df['label'].iloc[i: end_index])
            elif target_index == "first":
                labels.append(df['label'].iloc[i])
            else:  # last
                labels.append(df['label'].iloc[end_index])
        else:  # test
            if target_index == "n_to_n":
                labels.append(range(i, end_index))
            elif target_index == "first":
                labels.append(i)
            else:  # last
                labels.append(end_index)

    windows = np.array(windows).transpose(0, 2, 1)

    if shape_for == "lstm":
        windows = torch.tensor(windows).float()
    elif shape_for == "cnn":  # cnn
        windows = torch.transpose(torch.tensor(windows), 2, 1).float()
    else:  # for resnet
        freqs = []
        for window in windows:
            freqs.append(fast_fourier_transform(window)) # bring to frequency spectrum
        windows = torch.tensor(freqs).float()  # overwrite windows

    labels = np.array(labels)
    labels = torch.tensor(labels).long()

    return windows, labels


def fast_fourier_transform(sequence):
    """
    converts data from time to frequency
    :param sequence: the datasequence to transform
    :return: the transformed data in 1D representation (carefull with the dimensions)
    """
    # Perform Fourier transformation on each column
    fft_result = np.fft.fft(sequence, axis=0)

    # Get the magnitudes of the Fourier coefficients
    magnitudes = np.abs(fft_result)

    magnitudes = np.transpose(magnitudes)

    # Combine the columns into a single 1D output
    combined_output = magnitudes.ravel()  # Flattens the array into a 1D representation

    # Print the combined output
    return combined_output
