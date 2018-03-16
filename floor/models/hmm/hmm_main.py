import os
import numpy as np
from hmmlearn import hmm
from floor.data import dataset_loader
from test_tube import HyperOptArgumentParser, Experiment
from sklearn.externals import joblib


def main_trainer(hparams):
    print_params(hparams)

    exp = Experiment(name=hparams.tt_name,
                     debug=hparams.debug,
                     autosave=False,
                     description=hparams.tt_description,
                     save_dir=hparams.tt_save_path)

    exp.add_argparse_meta(hparams)

    # fit model
    val_scores = []
    best_score = 0
    for trial_nb in range(hparams.nb_trials):
        data = dataset_loader.IndividualSequencesData(hparams.data_path, y_labels=hparams.y_labels.split(','))
        X, Y, lengths = flatten_data(data.train_x_y)

        # fit
        model = hmm.GaussianHMM(n_components=hparams.nb_components, n_iter=hparams.nb_hmm_iters)
        model.fit(X, lengths)

        val_X, val_Y, lengths = flatten_data(data.val_x_y)
        Y_hat = model.predict(val_X, lengths)
        val_score = np.equal(Y_hat, val_Y).sum() / float(len(Y_hat))

        # save model
        if val_score > best_score:
            best_score = val_score
            save_model(model, hparams, exp, trial_nb)

        val_scores.append(val_score)

        exp.add_metric_row({'val_acc': val_score, 'trail_nb': trial_nb})

    mean_val_acc = np.mean(val_scores)
    exp.add_metric_row({'final_val_acc': mean_val_acc})
    exp.save()


def flatten_data(x_y_pairs):
    X = []
    Y = []
    lengths = []
    for trial in x_y_pairs:
        x, y = trial[0][0]
        lengths.append(x.shape[0])
        X.append(x.squeeze(1))
        Y.append(y.flatten())
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y, lengths


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

    # model params
    parser.add_argument('--nb_components', default=2, type=int)
    parser.add_argument('--nb_hmm_iters', default=100, type=int)
    parser.add_argument('--y_labels', default='indoors')

    parser.add_argument('--nb_trials', default=200, type=int)

    # path vars
    parser.add_argument('--data_path', default='/Users/waf/Developer/temp_floor/floor/data/in_out_classifier_data/data')
    parser.add_argument('--model_save_path', default='/Users/waf/Developer/temp_floor/floor/logs/weights/hmm')
    parser.add_argument('--tt_save_path', default='/Users/waf/Developer/temp_floor/floor/logs/training_logs/hmm')
    parser.add_argument('--tt_name', default='hmm_final_1')
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

