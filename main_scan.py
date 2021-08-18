import os
import sys
import copy
import json
import datetime

from models.params import params
import data.data as data
import utils

import talos
from tensorflow.keras.models import model_from_json
from keras_flops import get_flops

# Set seeds to ensure reproducability
import numpy as np
import tensorflow as tf
import random as python_random

seed = 0
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


def restore_models(scan):
    models = []
    for i, model_definition in enumerate(scan.saved_models):
        model = model_from_json(model_definition)
        model.set_weights(scan.saved_weights[i])
        model.compile('adam', 'categorical_crossentropy', metrics='accuracy')
        models.append(model)
    return models


def save_models(models, prefix):
    for i, model in enumerate(models):
        save_path = "{}/{}".format(prefix, 'model_{}'.format(i))
        model.save(save_path)


def run_tests(models, test_data, test_target, test_results):
    tmp_name = 'tmp/tmp_model'
    tmp_name_full = 'tmp/tmp_model.tflite'

    test_names = ['test_loss', 'test_accuracy',
                  'converted_accuracy', 'converted_size', 'num_params', 'flops']
    for name in test_names:
        test_results.insert(len(test_results.columns),
                            name, np.zeros(len(models)))

    for i, model in enumerate(models):
        # Add loss and accuracy
        loss, accuracy = model.evaluate(test_data, test_target)
        test_results['test_loss'][i] = loss
        test_results['test_accuracy'][i] = accuracy

        # Generate temporary converted model
        utils.convert_model(model, tmp_name, test_data, quantized=True)
        # Add converted_accuracy
        interpreter = utils.load_converted_model(tmp_name_full)
        converted_accuracy = utils.evaluate_converted_model(
            interpreter, test_data, test_target)
        test_results['converted_accuracy'][i] = converted_accuracy

        # Add converted_size
        model_size = os.stat(tmp_name_full).st_size
        test_results['converted_size'][i] = model_size

        # Remove temporary converted model
        os.remove(tmp_name_full)

        test_results['num_params'][i] = model.count_params()
        test_results['flops'][i] = get_flops(model)

    return test_results


def save_tests(test_results, prefix):
    test_results.to_csv(
        "{}/{}".format(prefix, 'test_results.csv'), sep=";", index_label="index")


def save_history(history, prefix):
    with open("{}/{}".format(prefix, 'history.json'), 'w') as file:
        file.write(json.dumps(history))


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Perform the scans
for model in (params if len(sys.argv) == 1 else sys.argv[1:]):
    data_configs = params[model]['config']
    if type(data_configs) is not list:
        data_configs = [data_configs]
    # Do scan for every config provided
    for data_config in data_configs:
        # Prepare data
        data.Dataset.set_config(data_config)
        dataset = data.Dataset()
        dataset.load()
        loaded_data, loaded_labels = dataset.get_tensors(as_tf_tensor=False)

        data_length = loaded_data.shape[0]
        train_length = int(data_config.train_split * data_length)

        # Split in train, validation and test data
        train_data = loaded_data[:train_length]
        train_target = loaded_labels[:train_length]

        test_data = loaded_data[train_length:]
        test_target = loaded_labels[train_length:]

        print("\n########################## Starting scan for model {} ##########################\n".format(model))
        scan_name = 'tmp/scan_{}_{}'.format(data_config.name, model)
        scan_dir = 'talos_scans/{}_{}/{}'.format(
            current_time, data_config.name, model)
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        if not os.path.isdir('talos_scans'):
            os.mkdir('talos_scans')

        scan = talos.Scan(x=train_data, y=train_target, x_val=test_data, y_val=test_target,
                        params=params[model]['params'], model=params[model]['model_func'],
                        experiment_name=scan_name)
        # Restore models
        restored_models = restore_models(scan)
        save_models(restored_models, scan_dir)
        # Perform tests
        test_results = run_tests(restored_models, test_data,
                                test_target, copy.copy(scan.data))
        save_tests(test_results, scan_dir)
        save_history(scan.round_history, scan_dir)

        print(test_results.to_string())

print('\n########################### Finished ###########################\n')
