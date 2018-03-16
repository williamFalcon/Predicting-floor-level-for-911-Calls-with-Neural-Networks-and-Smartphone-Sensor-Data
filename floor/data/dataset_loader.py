import numpy as np
import os
import pandas as pd


class TestData(object):

    def __init__(self, window_size, data_path, flatten_x=False, hmm_format=False, exp_name='_'):
        self.features = ['gps_vertical_accuracy', 'gps_horizontal_accuracy', 'gps_speed', 'rssi_strength', 'magnet_total']
        self.flatten_x = flatten_x
        self.hmm_format = hmm_format
        self.exp_name = exp_name
        self.window_size = window_size
        self.data_path = data_path
        self.nb_features = len(self.features)

    def test_gen(self):
        frames = self.load_frames(self.data_path)
        for df in frames:
            x_df, y_df = self.extract_features(df, self.features, 'indoors')

            X = x_df.as_matrix()
            Y = y_df.as_matrix()

            if not self.hmm_format:
                X, Y = self.create_window_features(X, Y, self.window_size, self.flatten_x)

            yield X, Y

    def create_window_features(self, X, Y, window_length=3, flatten_x=False):

        new_X = []
        new_Y = np.zeros((len(X) - window_length, 1))
        arr_i = 0
        side_size = int((window_length - 1) /2)
        for i in range(side_size, len(X)):
            i_start = i - side_size
            i_end = i + side_size + 1
            y_i = i
            dps = X[i_start:i_end, :]
            new_x = dps
            if flatten_x:
                new_x = dps.flatten()

            new_y = Y[y_i]
            if i_end >= len(X):
                break
            new_X.append(new_x)
            new_Y[arr_i] = new_y
            arr_i += 1

        return np.asarray(new_X), new_Y


    def extract_features(self, df, features, y_label):
        x_df = df[features]
        y_df = df[[y_label]]

        return x_df, y_df

    def load_frames(self, data_path):
        frames = []
        frames_names = []
        for file_name in os.listdir(data_path):
            if 'csv' in file_name and self.exp_name in file_name:
                in_path = '%s/%s' % (data_path, file_name)
                df = pd.read_csv(in_path)
                df = df.fillna(0)
                frames.append(df)
                frames_names.append(file_name)

        # add weather data
        for df in frames:
            df['weather_pressure'] = [100] * len(df)

        return frames


class IndividualSequencesData(object):
    """
    Each x,y pair is a discrete sequence of trials
    """
    def __init__(self, train_path, y_labels, train_split=0.8, data_limit=None, features=None, batch_size=3):
        self.train_path = train_path
        self.train_split = train_split
        self.data_limit = data_limit
        self.batch_size = batch_size

        self.features = features
        if self.features is None:
            self.features = ['gps_vertical_accuracy', 'gps_horizontal_accuracy', 'gps_speed', 'rssi_strength', 'magnet_total']

        self.y_labels = y_labels
        self.train_x_y, self.val_x_y = self.__load_train_val_data()

    @property
    def nb_features(self):
        return len(self.features)

    def nb_tng_batches(self):
        nb_dps = len(self.train_x_y)
        return nb_dps

    def nb_val_batches(self):
        nb_dps = len(self.val_x_y)
        return nb_dps

    def tng_generator(self, cuda_enabled, batch_first=False):
        data = self.train_x_y
        return self.generator(data, cuda_enabled, batch_first)

    def val_generator(self, cuda_enabled, batch_first=False):
        data = self.val_x_y
        return self.generator(data, cuda_enabled, batch_first)

    def generator(self, data, cuda_enabled, batch_first=False):

        for size_batches in data:
            for batch in size_batches:
                length = len(batch)

                x_dims = list(batch[0][0].shape)
                y_dims = list(batch[0][1].shape)
                x_dims[1] = length
                y_dims[1] = length

                if batch_first:
                    x_dims = [x_dims[1], x_dims[0], x_dims[2]]
                    y_dims = [y_dims[1], y_dims[0], y_dims[2]]

                x = np.zeros(shape=x_dims)
                y = np.zeros(shape=y_dims)
                for i, (x_i, y_i) in enumerate(batch):
                    if batch_first:
                        x[i, ...] = x_i[0, ...]
                        y[i, ...] = y_i[0, ...]
                    else:
                        x[:, i, ...] = x_i[:, 0, ...]
                        y[:, i, ...] = y_i[:, 0, ...]

                x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

                # convert to cuda tensors if gpu enabled
                if cuda_enabled:
                    x, y = x.cuda(), y.cuda()

                # prepare for pytorch
                x = Variable(x)
                y = Variable(y)

                yield x, y

    def __load_train_val_data(self):
        x_y_pairs = self.__load_files(self.train_path)

        np.random.shuffle(x_y_pairs)
        split_idx = int(self.train_split * len(x_y_pairs))
        x_y_train = x_y_pairs[0: split_idx]
        x_y_train = self.sort_batch(x_y_train, self.batch_size)

        x_y_val = x_y_pairs[split_idx:]
        x_y_val = self.sort_batch(x_y_val, self.batch_size)

        return x_y_train, x_y_val

    def sort_batch(self, x_y_pairs, batch_size):
        results = []

        # sort by seq len
        x_y_pairs = sorted(x_y_pairs, key=lambda x: x[0].shape[0])

        length = x_y_pairs[0][0].shape[0]
        to_batch = []
        for pair in x_y_pairs:
            # track shape until it changes, then start new batch
            pair_length = pair[0].shape[0]
            if pair_length == length:
                to_batch.append(pair)
            else:
                # start new batch and batch all the same sized pairs
                batched = self.__batch_in_chunks(to_batch, batch_size)
                results.append(batched)
                to_batch = [pair]
                length = pair_length

        # if have left over, make batches from those
        if len(to_batch) > 0:
            batched = self.__batch_in_chunks(to_batch, batch_size)
            results.append(batched)

        return results

    def __batch_in_chunks(self, x_y_pairs, batch_size):
        np.random.shuffle(x_y_pairs)
        batched = []
        for i in range(0, len(x_y_pairs), batch_size):
            i_end = i + batch_size
            batched.append(x_y_pairs[i: i_end])
        return batched



    def __load_files(self, path):
        frames = []
        files = os.listdir(path)
        for file in files:
            if '.csv' in file:
                full_path = os.path.join(path, file)
                df = pd.read_csv(full_path)
                x, y = self.__extract_features(df)
                frames.append((x, y))

                if self.data_limit and len(frames) > self.data_limit:
                    return frames

        return frames

    def __extract_features(self, df):
        x_df = df[self.features]
        y_df = df[self.y_labels]

        X = x_df.as_matrix()
        X = np.reshape(X, newshape=(X.shape[0], 1, X.shape[1]))

        Y = y_df.as_matrix()
        Y = np.reshape(Y, newshape=(Y.shape[0], 1, Y.shape[1]))

        return X, Y

class SequentialReadingsData(object):

    def __init__(self, window_size, data_path, flatten_x=False):
        features = ['gps_vertical_accuracy', 'gps_horizontal_accuracy', 'gps_speed', 'rssi_strength', 'magnet_total']
        self.nb_features = len(features)
        self.train_x, self.train_y, self.val_x, self.val_y = self.load_data(features, window_size, data_path, flatten_x)


    def load_data(self, features, window_size, data_path, flatten_x):
        df = self.merge_files(data_path)

        # shuffle data
        df = df.reindex(np.random.permutation(df.index))

        # extract features
        x_df, y_df = self.extract_features(df, features, 'indoors')

        X = x_df.as_matrix()
        Y = y_df.as_matrix()

        X, Y = self.create_window_features(X, Y, window_size, flatten_x)

        # calculate split values
        train_split = int(0.99 * len(X))

        # do splits
        X_train, Y_train = X[0: train_split], Y[0: train_split]
        X_val, Y_val = X[train_split:], Y[train_split:]

        return X_train, Y_train, X_val, Y_val

    def create_window_features(self, X, Y, window_length=3, flatten_x=False):

        new_X = []
        new_Y = np.zeros((len(X) - window_length, 1))
        arr_i = 0
        side_size = int((window_length - 1) / 2)
        for i in range(side_size, len(X)):
            i_start = i - side_size
            i_end = i + side_size + 1
            y_i = i
            dps = X[i_start:i_end, :]
            new_x = dps
            if flatten_x:
                new_x = dps.flatten()

            new_y = Y[y_i]
            if i_end >= len(X):
                break
            new_X.append(new_x)
            new_Y[arr_i] = new_y
            arr_i += 1

        return np.asarray(new_X), new_Y

    def extract_features(self, df, features, y_label):
        x_df = df[features]
        y_df = df[[y_label]]

        return x_df, y_df


    def merge_files(self, data_path):
        frames = []
        for file_name in os.listdir(data_path):
            if '.csv' in file_name:
                df = pd.read_csv(data_path + '/' + file_name)
                frames.append(df)

        joint = pd.concat(frames)

        # make the df go from 0...n sequentially
        joint = joint.reset_index()
        return joint