import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
from test_tube import HyperOptArgumentParser
from floor.util_scripts.test_model import test_model
import pandas as pd

"""
Tests a folder of models 
"""

def test_all_models(hparams):

    results = []
    for model_file in os.listdir(hparams.models_path):
        if '.pkl' in model_file or '.h5' in model_file:
            model_path = hparams.models_path + '/' + model_file
            acc = test_model(model_path=model_path, data_path=hparams.test_data_path)
            results.append({'name': model_file, 'acc': acc})
            print(model_file, acc)

    df = pd.DataFrame(results)
    df.to_csv(hparams.out_path)

if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='random_search')
    num = 11
    parser.add_argument('--models_path', type=str, default='/Users/waf/Developer/temp_floor/floor/logs/weights/lstm/lstm_final_{}'.format(num))
    parser.add_argument('--out_path', type=str, default='/Users/waf/Developer/temp_floor/floor/logs/weights/lstm/lstm_final_{}/final.csv'.format(num))

    parser.add_argument('--test_data_path', type=str, default='/Users/waf/Developer/temp_floor/floor/data/floor_prediction_test_data/data')
    hyperparams = parser.parse_args()

    test_all_models(hyperparams)
