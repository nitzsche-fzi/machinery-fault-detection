import sys

import tensorflow as tf
from keras_flops import get_flops
import matplotlib.pyplot as plt

import data.data as data
from models.params import params

# Set seeds for reproducibility
import numpy as np
import tensorflow as tf
import random as python_random

seed = 0
if len(sys.argv) == 3:
    seed = int(sys.argv[2])
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


if len(sys.argv) != 2:
    print("""
    Wrong number of arguments. Usage: python main_single.py <model_type>
    Possible values for <model_type> include:
    cnn     -   one layer convolutional network
    dscnn   -   two layer convolutional network with separable convolution
    wdcnn   -   three layer convolutional network
    dnn     -   two layer dense neural network
    """)
    sys.exit()

# Load data
data_config = data.config_preset_cwru
data_config.os_enabled = True
data_config.std_enabled = True

data.Dataset.set_config(data_config)
dataset = data.Dataset()
dataset.load()
loaded_data, loaded_labels = dataset.get_tensors()

data_length = loaded_data.shape[0]
train_length = int(data_config.train_split * data_length)

# Split in train, validation and test data
train_data = loaded_data[:train_length]
train_target = loaded_labels[:train_length]

test_data = loaded_data[train_length:]
test_target = loaded_labels[train_length:]

# Get the creation function
create_model = params[sys.argv[1]]['model_func']
# Define the params for the model (all ever possible values so that all models are compatible)
model_params = {
    'kernel_config': [(64, 1), (16, 1)],
    'filter_factor': 4,
    'has_batch_norm': True,
    'hidden_dense_size': 0,
    'size_first_layer': 8,
    'size_second_layer': 16,
    'batch_size': 128,
    'num_epochs': 50,
    'learning_rate': 0.004,
    'dropout': 0.1
}

# Create and train the model
history, model = create_model(train_data, train_target,
                              test_data, test_target, model_params)
# Evaluate the model with the test data
model.evaluate(test_data, test_target)

print("\nTraining finished\n")
model.summary()
print("Model FLOPS: {}".format(get_flops(model)))
plt.plot(history.history['val_accuracy'])
plt.show()
