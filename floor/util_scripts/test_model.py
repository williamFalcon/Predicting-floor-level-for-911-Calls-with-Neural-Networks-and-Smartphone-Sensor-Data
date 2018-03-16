from test_tube import HyperOptArgumentParser
import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
from sklearn.externals import joblib
from keras.models import load_model
from floor.data import TestData
import numpy as np

"""
Runs a model through the test data for IO classification

Example:
python find_max.py --exp_path /Users/waf/Dropbox/crackerjack/floor/test_tube/test_tube_data/final_1 --metric val_acc

"""

def test_model(model_path, data_path):
    # load model
    model = None
    is_keras_model = '.h5' in model_path
    is_lstm = 'lstm' in model_path

    flatten_x = not is_lstm

    # load pickle models
    if not is_keras_model:
        model = joblib.load(model_path)

    # load keras models
    else:
        model = load_model(model_path)

    is_hmm = 'hmm' in model_path

    data = TestData(window_size=3, data_path=data_path, flatten_x=flatten_x, hmm_format=is_hmm)
    test_gen = data.test_gen()

    n = 0.0
    correct = 0.0
    correct_inv = 0.0
    for X, Y in test_gen:
        Y = Y.flatten()

        # predict keras models
        if is_keras_model:
            Y_hat = model.predict(X)
            if is_lstm:
                Y_hat = Y_hat > 0.5
                Y_hat = Y_hat.reshape(len(Y_hat), 1).flatten()
            else:
                Y_hat = np.argmax(Y_hat, axis=1).reshape(len(Y_hat), 1).flatten()

        # predict sklearn models
        else:
            Y_hat = model.predict(X)
            Y_hat_inv = np.invert(Y_hat)

        n += len(Y_hat)
        nb_correct = np.equal(Y_hat, Y).sum()
        correct += nb_correct

        if is_hmm:
            correct_inv += np.equal(Y_hat_inv, Y).sum()

    total_acc = correct / n
    print('total_acc', total_acc)
    if is_hmm:
        print('inv_acc (possible hmm flip)', correct_inv / n)

    return total_acc


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--model_path', type=str, default='/Users/waf/Developer/temp_floor/floor/logs/weights/lstm/lstm_final_5/lstm_final_5_0_trial_0.h5')

    parser.add_argument('--test_data_path', type=str, default='/Users/waf/Developer/temp_floor/floor/data/floor_prediction_test_data/data')
    hyperparams = parser.parse_args()

    test_model(hyperparams.model_path, hyperparams.test_data_path)
