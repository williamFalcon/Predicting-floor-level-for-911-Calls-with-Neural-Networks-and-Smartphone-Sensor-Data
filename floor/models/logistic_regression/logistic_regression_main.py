from __future__ import print_function
import os
from floor.data import SequentialReadingsData
from test_tube import HyperOptArgumentParser, Experiment
from sklearn import linear_model
import numpy as np
from sklearn.externals import joblib


def main_trainer(hparams):
    print_params(hparams)

    exp = Experiment(name=hparams.tt_name,
                     debug=hparams.debug,
                     autosave=False,
                     description=hparams.tt_description,
                     save_dir=hparams.tt_save_path)

    exp.add_argparse_meta(hparams)

    # init data loader

    # fit model
    val_scores, train_scores = [], []
    best_score = 0
    for trial_nb in range(hparams.nb_trials):
        data = SequentialReadingsData(window_size=hparams.time_steps, data_path=hparams.data_path, flatten_x=True)

        clf = linear_model.LogisticRegression(C=hparams.C)
        clf.fit(data.train_x, data.train_y)

        train_score = clf.score(data.train_x, data.train_y)
        val_score = clf.score(data.val_x, data.val_y)

        # save model when we have a better one
        if val_score > best_score:
            best_score = val_score
            save_model(clf, hparams, exp, trial_nb)

        train_scores.append(train_score)
        val_scores.append(val_score)

        exp.add_metric_row({'val_acc': val_score, 'train_acc': train_score, 'trail_nb': trial_nb})

    mean_val_acc = np.mean(val_scores)
    mean_train_acc = np.mean(train_scores)
    exp.add_metric_row({'final_val_acc': mean_val_acc, 'final_train_acc': mean_train_acc})
    exp.save()

def save_model(clf, hparams, exp, trial_nb):
    # save model
    path = hparams.model_save_path + '/{}'.format(exp.name)
    if not os.path.isdir(path):
        os.makedirs(path)
    for f in os.listdir(path):
        if '.DS' not in f:
            os.remove(os.path.join(path, f))

    model_path = path + '/{}_{}_trial_{}.pkl'.format(exp.name, exp.version, trial_nb)

    joblib.dump(clf, model_path)


def print_params(hparams):
    # print params so we can see training
    print('-'*100, '\nTNG PARAMS:\n', '-'*100)
    for pk, pv in vars(hparams).items():
        print('{}: {}'.format(pk, pv))
    print('-'*100, '\n\n')

# build a wrapper around a tng function so we can use the correct gpu
# the optimizer passes in the hyperparams and the job index as arguments
# to the function to optimize
def parallelize_on_gpus(trial_params, job_index_nb):
    from time import sleep
    sleep(job_index_nb * 1)  # Time in seconds.

    GPUs = ['0', '1', '2', '4']
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs[job_index_nb]
    main_trainer(trial_params)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_opt_argument_list('--time_steps', default=3, options=[3, 5, 7, 9, 11], type=int, help='number of sequential readings', tunnable=False)

    # model params
    parser.add_opt_argument_list('--C', default=1e5, options=[1e5, 1e4, 1e3, 1e2, 1], type=int, tunnable=False)
    parser.add_argument('--nb_trials', default=200, type=int)

    # path vars
    parser.add_argument('--data_path', default='/Users/waf/Developer/temp_floor/floor/data/in_out_classifier_data/data')
    parser.add_argument('--model_save_path', default='/Users/waf/Developer/temp_floor/floor/logs/weights/logistic_regression')
    parser.add_argument('--tt_save_path', default='/Users/waf/Developer/temp_floor/floor/logs/training_logs/logistic_regression')
    parser.add_argument('--tt_name', default='logistic_regression_final_1')
    parser.add_argument('--tt_description', default='hyperopt')
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--local', default=True, type=bool)
    parser.add_json_config_argument('--config', default='/Users/waf/Developer/temp_floor/floor/logs/run_configs/local.json')

    hyperparams = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if hyperparams.local:
        main_trainer(hyperparams)
    else:
        hyperparams.optimize_parallel(parallelize_on_gpus, nb_trials=70, nb_parallel=4)

