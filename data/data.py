import tensorflow as tf
import numpy as np

from . import data_fns

class Config:
    name = ''
    type = ''
    train_split = 0.8

    classes = [
    ]

    class_names = [
    ]

    class_indices = []
    num_classes = 0

    base_path = ''

    # General data info
    CONST_FRAME_LENGTH = 512

    # Oversampling
    os_enabled = False
    os_window_len = 256
    os_stride = 16

    # Standardization settings
    std_enabled = False

    # Special parameters
    # The amount of data to load with each single file (so that dataset is balanced)
    load_slice = -1
    # The step length when loading the data (used for subsampling in e.g. the Paderborn dataset)
    load_step = 1


# Config preset for paper data: CWRU
config_preset_cwru = Config()
config_preset_cwru.type = "cwru_data"
config_preset_cwru.classes = [
    'Normal',
    'B_7',
    'B_14',
    'B_21',
    'IR_7',
    'IR_14',
    'IR_21',
    'OR_7',
    'OR_14',
    'OR_21'
]
config_preset_cwru.class_names = [
    'Normal',
    'Ball (7mm)',
    'Ball (14mm)',
    'Ball (21mm)',
    'Inner race (7mm)',
    'Inner race (14mm)',
    'Inner race (21mm)',
    'Outer race (7mm)',
    'Outer race (14mm)',
    'Outer race (21mm)',
]
config_preset_cwru.num_classes = 10
config_preset_cwru.base_path = 'dataset_cwru'
config_preset_cwru.load_slice = 120000

config_preset_cwru.CONST_FRAME_LENGTH = 1024
config_preset_cwru.os_window_len = 512
config_preset_cwru.os_stride = 32


# Config preset for paper data: Paderborn
config_preset_paderborn = Config()
config_preset_paderborn.type = "paderborn_data"
config_preset_paderborn.classes = [
    "Normal",
    "Inner_Race",
    "Outer_Race"
]
config_preset_paderborn.class_names = [
    'Normal',
    'Inner race',
    'Outer race',
]
config_preset_paderborn.num_classes = 3
config_preset_paderborn.base_path = 'dataset_paderborn'
config_preset_paderborn.load_slice = 256000
config_preset_paderborn.load_step = 4

config_preset_paderborn.CONST_FRAME_LENGTH = 2048
config_preset_paderborn.os_window_len = 1024
config_preset_paderborn.os_stride = 64


class Dataset:
    config = Config()


    @staticmethod
    def set_config(config):
        Dataset.config = config


    def load(self):
        '''
        Load the data from file using the config set in the
        constructor.
        '''
        self.dataset_raw, self.labels = data_fns.dataset_load(self.config)
        # Process the dataset with standardization
        self.dataset_processed = data_fns.dataset_process(self.dataset_raw, self.config)
        self.dataset_final = np.expand_dims(self.dataset_processed, axis=2)


    def get_tensors(self, shuffle=True, as_tf_tensor=True):
        '''
        Convert the loaded dataset to tensorflow tensors. If 'shuffle'
        is True, the dataset and labels get randomly shuffled.
        '''
        one_hot_labels = tf.keras.utils.to_categorical(self.labels, num_classes=self.config.num_classes)
        if shuffle:
            dataset, labels = data_fns.dataset_shuffle(self.dataset_final, one_hot_labels)
        else:
            dataset = self.dataset_final
            labels = one_hot_labels

        dataset = dataset.astype(np.float32)
        if as_tf_tensor:
            dataset_tensor = tf.constant(dataset)
            labels_tensor = tf.constant(labels)
            return dataset_tensor, labels_tensor
        else:
            return dataset, labels
