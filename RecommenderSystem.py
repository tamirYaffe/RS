import os
import numpy as np
import csv
import sklearn as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load(filepath):
    """
    Load the csv data files into a structure of our choice dataframe or matrix.
    :param filepath: path for the csv file(should not be local).
    :return: a generator for the data structure of the loaded file.
    """
    # print(filepath)
    def load_gen(filepath):
        with open(filepath, newline='') as org_file:
            l_idx = 0
            reader = csv.reader(org_file)
            for row_ in reader:
                if l_idx != 0:
                    yield row_
                l_idx += 1
    all_data_gen = load_gen(filepath)
    return all_data_gen


def RMSE(true_ranks, predicted_ranks):
    """
    Calculate the root mean square error for the ranking.
    :param true_ranks: actual ranking.
    :param predicted_ranks: the model predicted ranking.
    :return:
    """
    return mean_squared_error(true_ranks, predicted_ranks, squared=False)
    # print(true_ranks, predicted_ranks)



def accuracyEval(true_ranks, predicted_ranks):
    """
    Calculate the ??? for the ranking.
    :param true_ranks: actual ranking.
    :param predicted_ranks: the model predicted ranking.
    :return:
    """
    # todo: decide of a second evaluation method.
    print(true_ranks, predicted_ranks)
    pass


# todo: understand the input for this function
def PredictRating(model, data_to_predict):
    print(model, data_to_predict)
    pass

class BaseSVDModel:
    def __init__(self, latent_features_size, user_size, items_size, ranking_mean):
        self.latent_features_size = latent_features_size
        self.user_size = user_size
        self.item_size = items_size

        self._Q = np.random.rand((self.item_size, self.latent_features_size))
        self.BI = np.random.rand((self.item_size, self.latent_features_size))

        self._P = np.random.rand((self.user_size, self.latent_features_size))
        self.BU = np.random.rand((self.user_size, self.latent_features_size))

        self.MU = ranking_mean

        self.gamma = 0.005
        self.lamda = 0.02

    def predict(self, user_id, item_id):
        pass

def get_sizes(path_to_training):
    """
    :param path_to_training: full path to training file
    :return: number of unique items, number of unique users, the mean ranking
    """
    generator = load(path_to_training)
    single_record = next(generator)
    ranking_sum = 0
    rankings_cnt = 0
    business_id = {}
    user_id = {}
    while single_record != None:
        user_id[single_record[1]] = 1
        business_id[single_record[2]] = 1
        ranking_sum += float(single_record[3])
        rankings_cnt += 1
        try:
            single_record = next(generator)
        except StopIteration as e:
            break

    return len(business_id.keys()), len(user_id.keys()), ranking_sum/rankings_cnt


def split_and_save_train_validation(train_path, train_split_path, valid_split_path, validation_percent=0.3):
    if os.path.exists(train_split_path) or os.path.exists(valid_split_path):
        os.remove(train_split_path)
        os.remove(valid_split_path)

    train_file = csv.writer(open(train_split_path, 'a', newline=''))
    val_file = csv.writer(open(valid_split_path, 'a', newline=''))

    train_size = 0
    valid_size = 0
    generator = load(train_path)
    single_record = next(generator)
    while single_record != None:
        if valid_size <= train_size * validation_percent:
            if np.random.rand((1))[0] > validation_percent:
                val_file.writerow(single_record)
                valid_size += 1
        else:
            train_file.writerow(single_record)
            train_size += 1
        try:
            single_record = next(generator)
        except StopIteration as e:
            break
    print('val:{}\ntrain:{}'.format(valid_size, train_size))



def TrainBaseModel(latent_features_size, train_data_path):
    """
    Implement of the basic model described in Recommender Systems Handbook, chapter 3, section 3.3
    :param latent_features_size: number of features for the model.
    :param train_data_path: the path for the training data of the ranking, created via the load function.
    :return: trained model.
    """

    # print(latent_features_size, len(train_data_path))
    items_cnt, users_cnt, ranking_mean = get_sizes(train_data_path)
    model = {}

    # split train_data into train and validation.
    train_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['train_split.csv'])
    valid_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['valid_split.csv'])
    split_and_save_train_validation(train_data_path,
                                    train_split_path=train_split_path,
                                    valid_split_path=valid_split_path,
                                    validation_percent=0.3)

    # randomly initialize U, b_u, b_i, p_u, q_i.
    model = BaseSVDModel(latent_features_size, user_size=users_cnt, items_size=items_cnt, ranking_mean=ranking_mean)

    # todo: iterate over the ranking in train_data and for each:
    #       1. calculate the error E_ui.
    #       2. update U, b_u, b_i, p_u, q_i.


    # todo: calculate RMSE over the validation, stop when is larger from prev iteration.
    # todo: improve RMSE by updating γ and λ.(???)
    return model


def TrainImprovedModel():
    """
    Train an improved model from the prev SVD model, by the paper in https://dl.acm.org/doi/pdf/10.1145/1401890.1401944.
    """
    pass


def TrainContentModel():
    pass


def TrainHybridModel():
    pass


if __name__ == '__main__':
    train_data_path = "Data/userTrainDataSmall.csv"
    train_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['train_split.csv'])
    valid_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['valid_split.csv'])
    split_and_save_train_validation(train_data_path,
                                    train_split_path=train_split_path,
                                    valid_split_path=valid_split_path,
                                    validation_percent=0.3)

