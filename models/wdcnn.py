import tensorflow as tf
import tensorflow.keras as keras

import data.data as data

'''
Wide Deep Convolutional Neural Network (WDCNN) with 3 layers. Performs really well so far.
'''

def create_model(train_data, train_target, val_data, val_target, params):
    num_output_classes = data.Dataset.config.num_classes
    input_shape = train_data[0].shape

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape, name='input'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(filters=8*params['filter_factor'], kernel_size=params['kernel_config'][0], strides=params['kernel_config'][1], padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))

    if params['has_batch_norm']:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=16*params['filter_factor'], kernel_size=(3, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))

    if params['has_batch_norm']:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=32*params['filter_factor'], kernel_size=(3, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(keras.layers.Flatten())
    if params['hidden_dense_size'] > 0:
        if params['has_batch_norm']:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(params['hidden_dense_size'], activation='relu'))

    model.add(keras.layers.Dropout(params['dropout']))
    model.add(keras.layers.Dense(num_output_classes, activation='softmax'))

    # Create the checkpoint callback
    checkpoint_filepath = 'tmp/checkpoint'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    history = model.fit(train_data, train_target, validation_data=(val_data, val_target),
                        epochs=params['num_epochs'], batch_size=params['batch_size'], callbacks=[checkpoint_callback])

    # Load the best performing checkpoint into the model
    model.load_weights(checkpoint_filepath)

    return history, model
