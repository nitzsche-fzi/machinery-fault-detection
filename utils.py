import tensorflow as tf
import numpy as np

'''
This module is for miscellaneous utility functions
for handling models.
'''

def convert_model(model, filename, example_data=None, quantized=False):
    '''
    Converts the given 'model' to the TF Lite format and saves
    it to a file as 'filename.tflite'. If the model should be
    8 bit quantized, the 'quantized' argument must be set to 'True'.
    The given 'example_data' gets used for the representative dataset
    when doing quantization (it can be None if 'quantized' is False).
    '''
    # Quantize and convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantized:
        def dataset_gen():
            num_calib_steps = 200
            for i in range(num_calib_steps):
                yield [example_data[i:i+1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = dataset_gen

    tflite_quant_model = converter.convert()

    with tf.io.gfile.GFile(filename + '.tflite', 'wb') as f:
        f.write(tflite_quant_model)


def load_converted_model(file_path):
    '''
    Load the '.tflite' model from 'file_path' and returns an interpreter.
    '''
    interpreter = tf.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()
    return interpreter


def evaluate_converted_model(interpreter, test_data, test_target):
    '''
    Evaluates the model using the given 'interpreter' and the
    'data' and 'labels'. The labels should be a number representing
    the target category instead of being "one-hot" encoded.
    Returns the accuracy.
    '''
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Reverse one hot encoding
    test_target = tf.argmax(tf.cast(test_target, tf.float32), axis=1)

    prediction_digits = []
    print('Invoke...')
    for sample in test_data:
        sample = tf.expand_dims(sample, axis=0)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    accurate_count = 0
    print('Evaluate...')
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_target[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    return accuracy


def load_and_evaluate_model(file_path, test_data, test_target):
    '''
    Loads and evaluates the model in the given 'file_path'
    and prints the calculated accuracy. The data used is the
    one that also gets used for training.
    '''
    interpreter = load_converted_model(file_path)
    accuracy = evaluate_converted_model(interpreter, test_data, test_target)
    print("Evaluated accuracy: ", accuracy)
