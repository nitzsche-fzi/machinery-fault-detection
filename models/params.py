# Normal models
import models.cnn as cnn
import models.dscnn as dscnn
import models.wdcnn as wdcnn
import models.dnn as dnn

import copy
import data.data as data

config_cwru = copy.copy(data.config_preset_cwru)
config_cwru.name = 'config_cwru'
config_cwru.os_enabled = True

config_paderborn = copy.copy(data.config_preset_paderborn)
config_paderborn.name = 'config_paderborn'
config_paderborn.os_enabled = True

params = {
    'cnn': {
        'model_func': cnn.create_model,
        'params': {
            'kernel_config': [[(128, 1), (32, 1)], [(64, 1), (16, 1)], [(32, 1), (8, 1)]],
            'filter_factor': [4, 5, 6],
            'has_batch_norm': [True],
            'batch_size': [128],
            'num_epochs': [50],
            'learning_rate': [0.004],
            'dropout': [0.1]
        },
        'config': [config_cwru, config_paderborn]
    },
    'dscnn': {
        'model_func': dscnn.create_model,
        'params': {
            'has_batch_norm': [True],
            'kernel_config': [[(128, 1), (32, 1)], [(64, 1), (16, 1)], [(32, 1), (8, 1)]],
            'filter_factor': [2, 2.5, 3],
            'batch_size': [128],
            'num_epochs': [50],
            'learning_rate': [0.004],
            'dropout': [0.1]
        },
        'config': [config_cwru, config_paderborn]
    },
    'wdcnn': {
        'model_func': wdcnn.create_model,
        'params': {
            'kernel_config': [[(128, 1), (32, 1)], [(64, 1), (16, 1)], [(32, 1), (8, 1)]],
            'filter_factor': [0.75, 1],
            'has_batch_norm': [True],
            'hidden_dense_size': [0, 20],
            'batch_size': [128],
            'num_epochs': [50],
            'learning_rate': [0.004],
            'dropout': [0.1]
        },
        'config': [config_cwru, config_paderborn]
    },
    'dnn': {
        'model_func': dnn.create_model,
        'params': {
            'has_batch_norm': [True],
            'size_first_layer': [8, 16],
            'size_second_layer': [0, 8, 16],
            'batch_size': [128],
            'num_epochs': [20],
            'learning_rate': [0.004],
            'dropout': [0.1],
        },
        'config': [config_cwru, config_paderborn]
    }
}
