from __future__ import print_function
import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
from floor.data import SequentialReadingsData
from test_tube import HyperOptArgumentParser, Experiment
import numpy as np
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


def main_trainer(hparams):
    print_params(hparams)
    full_exp = Experiment(name=hparams.tt_name+'_overall',
                     debug=hparams.debug,
                     autosave=False,
                     description=hparams.tt_description,
                     save_dir=hparams.tt_save_path)

    full_exp.add_argparse_meta(hparams)

    # fit model
    val_scores, train_scores = [], []
    best_acc = 0
    best_loss = 0
    best_trial_nb = 0
    for trial_nb in range(hparams.nb_trials):
        exp = Experiment(name=hparams.tt_name,
                         debug=hparams.debug,
                         autosave=False,
                         description=hparams.tt_description,
                         save_dir=hparams.tt_save_path)

        exp.add_argparse_meta(hparams)
        data = SequentialReadingsData(window_size=hparams.time_steps, data_path=hparams.data_path)

        val_loss, val_acc, history = fit_feedforward(hparams, exp, trial_nb, data)
        log_history(history.history, exp)

        exp.add_metric_row({'final_val_acc': val_acc, 'final_train_acc': val_loss})
        exp.save()

        full_exp.add_metric_row({'val_acc': val_acc, 'val_loss': val_loss, 'trial_nb': trial_nb})

        # save model when we have a better one
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            best_trial_nb = trial_nb

        val_scores.append(val_acc)

    mean_val_acc = np.mean(val_scores)
    full_exp.add_metric_row({'final_val_acc': mean_val_acc, 'best_val_loss': best_loss, 'best_val_acc': best_acc, 'best_trial_nb': best_trial_nb})
    full_exp.save()


def log_history(history, exp):
    losses = history['loss']
    accs = history['acc']
    for epoch_nb, loss, acc in zip(list(range(len(losses))), losses, accs):
        exp.add_metric_row({'train_acc': acc, 'epoch_nb': epoch_nb, 'train_loss': loss})


def fit_feedforward(hparams, exp, trial_nb, data):

    model = Sequential()

    model.add(LSTM(hparams.nb_rnn_units_l1, input_shape=(hparams.time_steps, data.nb_features), return_sequences=True,
                   dropout=hparams.drop_rate, recurrent_dropout=hparams.drop_rate))
    # model.add(LSTM(hparams.nb_rnn_units_l2, return_sequences=True, dropout=hparams.drop_rate,
    #                recurrent_dropout=hparams.drop_rate))
    model.add(LSTM(2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss=hparams.loss,
                  optimizer=Adam(hparams.learning_rate),
                  metrics=['accuracy'])

    path = hparams.model_save_path + '/{}'.format(exp.name)
    if not os.path.isdir(path):
        os.makedirs(path)

    model_path = path + '/{}_{}_trial_{}.h5'.format(exp.name, exp.version, trial_nb)

    # save model whenever we get a better validation score
    checkpoint = ModelCheckpoint(model_path,
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=40,
                              verbose=1, mode='auto')

    history = model.fit(data.train_x, data.train_y,
                        batch_size=hparams.batch_size, nb_epoch=hparams.nb_epochs,
                        callbacks=[checkpoint, earlystop],
                        verbose=1)

    scores = model.evaluate(data.val_x, data.val_y, verbose=0)
    val_loss, val_acc = scores
    return val_loss, val_acc, history


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
    parser.add_opt_argument_list('--nb_rnn_units_l1', default=128, options=[50, 64, 128], type=int, tunnable=True)
    parser.add_opt_argument_list('--nb_rnn_units_l2', default=9, options=[50, 64, 128], type=int, tunnable=True)
    parser.add_opt_argument_list('--drop_rate', default=0.2, options=[0.2, 0.5, 0.7], type=float, tunnable=True)
    parser.add_opt_argument_list('--learning_rate', default=0.004, options=[0.004, 0.005, 0.006, 0.007], type=float, tunnable=True)
    parser.add_opt_argument_list('--batch_size', default=128, options=[128, 256], type=int, tunnable=False)

    # model params
    parser.add_argument('--nb_trials', default=10, type=int)
    parser.add_argument('--nb_epochs', default=600, type=int)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--loss', default='binary_crossentropy', type=str)

    # path vars
    parser.add_argument('--data_path', default='/Users/waf/Developer/temp_floor/floor/data/in_out_classifier_data/data')
    parser.add_argument('--model_save_path', default='/Users/waf/Developer/temp_floor/floor/logs/weights/lstm')
    parser.add_argument('--tt_save_path', default='/Users/waf/Developer/temp_floor/floor/logs/training_logs/lstm')
    parser.add_argument('--tt_name', default='lstm_final_11')
    parser.add_argument('--tt_description', default='hyperopt')
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--local', default=True, type=bool)
    parser.add_json_config_argument('--config', default='/Users/waf/Developer/temp_floor/floor/logs/run_configs/local.json')

    hyperparams = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if hyperparams.local:
        main_trainer(hyperparams)
    else:
        hyperparams.optimize_parallel(parallelize_on_gpus, nb_trials=36, nb_parallel=4)

