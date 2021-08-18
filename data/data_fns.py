import numpy as np
import pandas as pd
from scipy.io import loadmat
import os


'''
This module provides all of the data loading and manipulation functions.
The finished dataset always looks like this:

    shape = [num_frames, frame_length, feature_length, num_channels]

In our case, the feature_length and num_channels are always 1.
'''

def dataframe_load_cwru(file_path):
    '''
    Loads the data from the given 'file_path' and converts it from the '.mat'
    format. Returns a pandas.DataFrame with the data in a one dimensional frame.
    '''
    mat_dict = loadmat(file_path)
    mat_data = mat_dict[list(mat_dict.keys())[3]]
    # Little hack here, some files have one field more before the one used
    if len(mat_data) == 1:
        mat_data = mat_dict[list(mat_dict.keys())[4]]
    return pd.DataFrame(mat_data)


def dataframe_load_paderborn(file_path):
    '''
    Loads the data from the figen 'file_path'. The path must contain a '.csv' file.
    '''
    df = pd.read_csv(file_path, header=None)
    return df


def class_oversample(class_data, config):
    '''
    Slices the data into to parts with length of 'window_length'
    and offsets of 'stride'. This results in an oversampling and
    more data to work with.
    '''
    # Calculate the number of frames
    num_windows = (class_data.shape[1] - config.os_window_len) // config.os_stride + 1
    if num_windows == 0:
        print("Error: window length is greater than data length.")
        return
    # Slice the frames
    os_frames_list = list()
    for class_frame in class_data:
        for i in range(num_windows):
            p = i * config.os_stride
            os_frame = class_frame[p:p+config.os_window_len]
            os_frames_list.append(os_frame)
    # Fuse the frames
    os_class_data = np.stack(os_frames_list)
    return os_class_data


def class_load(class_name, config):
    '''
    Loads the data of one class present in the config. It takes in the
    'class_name' that should be loaded, constructs the path and loads
    the data.
    Returns: a numpy ndarray with shape [num_frames, frame_length, 1].
    '''
    data_frames = list()

    directory_path = os.path.join(config.base_path, class_name)
    # Get all files in the directory of the class
    files_list = os.listdir(directory_path)
    for file_name in files_list:
        file_path = os.path.join(directory_path, file_name)
        if config.type == "cwru_data":
            loaded_df = dataframe_load_cwru(file_path)
        elif config.type == "paderborn_data":
            loaded_df = dataframe_load_paderborn(file_path)
        else:
            return None
        # Subsample if needed
        data_frame = loaded_df.to_numpy()[:config.load_slice][0::config.load_step]
        # Reshape into equal sized frames
        data_frame = data_frame[
            :(len(data_frame)//config.CONST_FRAME_LENGTH) * config.CONST_FRAME_LENGTH
        ].reshape((-1, config.CONST_FRAME_LENGTH))
        data_frame = np.expand_dims(data_frame, axis=-1)
        data_frames.append(data_frame)

    return np.concatenate(data_frames, axis=0)


def dataset_shuffle(dataset, labels, seed=42):
    '''
    Randomly shuffles the rows of the given data and labels.
    '''
    # Create np.array with the indices and shuffle them
    indices = np.arange(dataset.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    # Gather the dataset and labels based on the shuffled indices
    shuffled_dataset = np.take(dataset, indices, axis=0)
    shuffled_labels = np.take(labels, indices, axis=0)

    return shuffled_dataset, shuffled_labels


def dataset_load(config):
    '''
    Loads the complete dataset specified in the 'config'. Also all
    wanted preprocessing steps (like STFT conversion, oversampling,
    filtering, interpolating) get done here.
    '''
    data_classes = list()
    labels = list()

    # This is for keeping track of what indices the classes have
    class_indices = list()
    class_index = 0

    for label, class_name in enumerate(config.classes, start=0):
        data_class = class_load(class_name, config)
        class_indices.append(class_index)
        if config.os_enabled:
            # Oversample if OS is enabled
            data_class = class_oversample(data_class, config)
        data_classes.append(data_class)
        class_index += data_class.shape[0]
        labels.extend([label] * data_class.shape[0])

    dataset = np.concatenate(data_classes, axis=0)
    labels = np.stack(labels, axis=0)
    config.class_indices = class_indices

    return dataset, labels


def dataset_process(dataset, config):
    '''
    Process the dataset with the desired methods.
    (In this case only standardization is done)
    '''
    if config.std_enabled:
        dataset = dataset_standardize(dataset)

    return dataset


def dataset_standardize(dataset, use_bias=True, use_std_dev=True):
    '''
    Normalizes the dataset to a variance of 1 and a
    mean of 0.
    '''
    dataset = dataset.astype(np.float32)
    # Normalize bias for each channel
    for i in range(dataset.shape[-1]):
        # Calculate mean and standard deviation
        mean = np.mean(dataset[...,i])
        std_dev = np.std(dataset[...,i])
        # Apply the values
        if use_bias:
            dataset[...,i] -= mean
        if use_std_dev:
            dataset[...,i] /= std_dev

    return dataset


def save_to_file(frame, filename):
    '''
    Saves the given 'frame' as a .h file to 'filename'.h.
    The generated code is not pretty, but can be autoformatted
    after saving.
    The frame should have the shape [samples_per_frame, features_per_frame, num_channels].
    '''
    frame_shape = frame.shape
    num_written_floats = 0
    with open(filename + ".h", 'w') as file:
        file.write ('#ifndef INPUT_DATA_\n')
        file.write ('#define INPUT_DATA_\n\n')
        # Write header
        file.write("const float {0}[1][{1}][{2}][{3}]".format(
            filename, frame_shape[0], frame_shape[1], frame_shape[2]
        ))

        file.write("{")
        file.write("{")
        for column in frame:
            file.write("{")
            for row in column:
                file.write("{")
                for channel in row:
                    file.write(str(float(channel)) + ", ")
                    num_written_floats += 1
                file.write("\n")
                file.write("},")
            file.write("},")
        file.write("}")
        file.write("};\n")
        file.write("const unsigned int " + filename + "_len = "
                   + str(num_written_floats) + ";\n\n")
        file.write ('#endif\n')
