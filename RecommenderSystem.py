import os
from abc import ABC
import numpy as np
import csv
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import math

np.random.seed(420)


def load(filepath):
    """
    Load the csv data files into a structure of our choice dataframe or matrix.
    :param filepath: path for the csv file(should not be local).
    :return: a generator for the data structure of the loaded file.
    """

    # print(filepath)
    def load_gen(path):
        with open(path, newline='') as org_file:
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
    acc = sum(list(map(lambda x, y: abs(x - y) < 1, true_ranks, predicted_ranks))) / len(true_ranks)
    print("validation acc:{}".format(acc))
    return mean_squared_error(true_ranks, predicted_ranks, squared=False)
    # print(true_ranks, predicted_ranks)


def accuracyEval(true_ranks, predicted_ranks):
    """
    Calculate the Root Mean Squared Log Error (RMSLE) for the ranking.
    :param true_ranks: actual ranking.
    :param predicted_ranks: the model predicted ranking.
    :return:
    """
    if sum([x < 0 for x in predicted_ranks]) > 0:
        return -1
    rmsle = np.sqrt(mean_squared_log_error(true_ranks, predicted_ranks))
    return rmsle


# todo: understand the input for this function
def PredictRating(model, data_to_predict):
    print(model, data_to_predict)
    pass


class ABSModelInterface(ABC):
    def predict(self, user_id, item_id):
        pass

    def correction(self, error, user_id, item_id):
        pass


class SVDPlusModel(ABSModelInterface):
    def __init__(self, latent_features_size, users_ids, items_ids, ranking_mean, ru_dict, lamda, gamma1, gamma2):
        self.error_threshold = 0.01
        self.latent_features_size = latent_features_size
        self.user_size = len(users_ids)
        self.item_size = len(items_ids)
        #
        self.Q = {key: np.random.rand(latent_features_size) for key in items_ids}
        self.BI = self.BU = {key: np.random.rand(1) / 100 for key in items_ids}

        self.P = {key: np.random.rand(latent_features_size) for key in users_ids}
        self.BU = {key: np.random.rand(1) / 100 for key in users_ids}
        #
        self.MU = ranking_mean
        self.Y = {key: np.random.randn(latent_features_size) for key in items_ids}
        self.RU = ru_dict
        #
        self.lamda = lamda
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def predict(self, user_id, item_id):
        # todo: change dict call to get(, 0)
        user_latent_vec = self.P.get(user_id, np.zeros(self.latent_features_size))
        item_latent_vec = self.Q.get(item_id, np.zeros(self.latent_features_size))
        Bi = self.BI.get(item_id, 0)
        Bu = self.BU.get(user_id, 0)

        Ru = self.RU.get(user_id, [])
        if len(Ru) == 0:
            pred_val = self.MU + Bi

        else:
            y_i = np.array(list(map(lambda x: self.Y[x], self.RU[user_id]))).sum()
            # y_i = np.array([self.Y[it_id] for it_id in self.RU[user_id]]).sum()
            pred_val = self.MU + Bi + Bu + np.dot(item_latent_vec, (user_latent_vec + y_i / math.sqrt(len(Ru))))

        pred_val = pred_val[0]
        if pred_val > 5:
            pred_val = 5
        if pred_val < 1:
            pred_val = 1

        return pred_val

    def correction(self, error, user_id, item_id):
        if abs(error) < self.error_threshold:
            return
        y_i = np.array(list(map(lambda x: self.Y[x], self.RU[user_id]))).sum()
        # y_i = np.array([self.Y[it_id] for it_id in self.RU[user_id]]).sum()

        self.BU[user_id] = self.BU[user_id] + self.lamda * (error - self.gamma1 * self.BU[user_id])
        self.BI[item_id] = self.BI[item_id] + self.lamda * (error - self.gamma1 * self.BI[item_id])

        temp_qi = self.Q[item_id].copy()
        self.Q[item_id] = self.Q[item_id] + self.lamda * (
                    error * (self.P[user_id] + y_i / math.sqrt(len(self.RU[user_id]))) - self.gamma2 * self.Q[item_id])
        self.P[user_id] = self.P[user_id] + self.lamda * (error * temp_qi - self.gamma2 * self.P[user_id])

        for y_item_id in self.RU[user_id]:
            self.Y[y_item_id] = self.Y[y_item_id] + self.lamda * \
                                (error / math.sqrt(len(self.RU[user_id])) * temp_qi - self.gamma2 * self.Y[y_item_id])


class BaseSVDModel(ABSModelInterface):
    def __init__(self, latent_features_size, users_ids, items_ids, ranking_mean, lamda, gamma):
        self.error_threshold = 0.01
        self.latent_features_size = latent_features_size
        self.user_size = len(users_ids)
        self.item_size = len(items_ids)

        self.Q = {key: np.random.rand(latent_features_size) for key in items_ids}
        self.BI = self.BU = {key: np.random.rand(1) / 100 for key in items_ids}

        self.P = {key: np.random.rand(latent_features_size) for key in users_ids}
        self.BU = {key: np.random.rand(1) / 100 for key in users_ids}

        self.MU = ranking_mean

        self.lamda = lamda
        self.gamma = gamma

    def predict(self, user_id, item_id):
        user_latent_vec = self.P.get(user_id, np.zeros(self.latent_features_size))
        item_latent_vec = self.Q.get(item_id, np.zeros(self.latent_features_size))
        Bi = self.BI.get(item_id, 0)
        Bu = self.BU.get(user_id, 0)

        pred_val = self.MU + Bi + Bu + np.dot(user_latent_vec, item_latent_vec)
        pred_val = pred_val[0]
        if pred_val > 5:
            pred_val = 5
        if pred_val < 1:
            pred_val = 1

        return pred_val

    def correction(self, error, user_id, item_id):
        if abs(error) < self.error_threshold:
            return
        self.BU[user_id] = self.BU[user_id] + self.lamda * (error - self.gamma * self.BU[user_id])
        self.BI[item_id] = self.BI[item_id] + self.lamda * (error - self.gamma * self.BI[item_id])

        temp_qi = self.Q[item_id].copy()
        self.Q[item_id] = self.Q[item_id] + self.lamda * (error * self.P[user_id] - self.gamma * self.Q[item_id])
        self.P[user_id] = self.P[user_id] + self.lamda * (error * temp_qi - self.gamma * self.P[user_id])


def get_unique_users_and_items(path_to_training):
    """
    :param path_to_training: full path to training file
    :return: number of unique items, number of unique users, the mean ranking
    """
    ru_dict = {}
    generator = load(path_to_training)
    single_record = next(generator)
    ranking_sum = 0
    rankings_cnt = 0
    business_id = {}
    user_id = {}
    while single_record is not None:
        curr_user_id = single_record[1]
        curr_item_id = single_record[2]
        curr_ranking = single_record[3]
        user_set = ru_dict.get(curr_user_id, set())
        user_set.add(curr_item_id)
        ru_dict[curr_user_id] = user_set
        user_id[curr_user_id] = 1
        business_id[curr_item_id] = 1
        ranking_sum += float(curr_ranking)
        rankings_cnt += 1
        try:
            single_record = next(generator)
        except StopIteration:
            break

    return list(business_id.keys()), list(user_id.keys()), ranking_sum / rankings_cnt, ru_dict


def split_and_save_train_validation(train_path, train_split_path, valid_split_path, validation_percent=0.2):
    if os.path.exists(train_split_path) or os.path.exists(valid_split_path):
        os.remove(train_split_path)
        os.remove(valid_split_path)

    train_file = csv.writer(open(train_split_path, 'a', newline=''))
    val_file = csv.writer(open(valid_split_path, 'a', newline=''))

    train_size = 0
    valid_size = 0
    generator = load(train_path)
    single_record = next(generator)
    while single_record is not None:
        if valid_size <= train_size * validation_percent:
            if np.random.rand(1)[0] > validation_percent:
                val_file.writerow(single_record)
                valid_size += 1
        else:
            train_file.writerow(single_record)
            train_size += 1
        try:
            single_record = next(generator)
        except StopIteration:
            break
    print('val:{}\ntrain:{}'.format(valid_size, train_size))


def train_model(model, train_gen):
    """
    trains a model which implements the ABSModelInterface class.
    this function can and is used for single epoch training round.
    :param model: the model to train.
    :param train_gen: a generator which iterates through all records to train on. throws StopIteration when finished.
    """
    predicted_rankings = []
    true_rankings = []
    single_record = next(train_gen)
    while single_record is not None:
        curr_user_id = single_record[1]
        curr_item_id = single_record[2]
        curr_rank = float(single_record[3])
        curr_prediction = model.predict(curr_user_id, curr_item_id)
        error = curr_rank - curr_prediction
        model.correction(error=error, user_id=curr_user_id, item_id=curr_item_id)
        true_rankings.append(curr_rank)
        predicted_rankings.append(curr_prediction)
        try:
            single_record = next(train_gen)
        except StopIteration:
            break
    acc = sum(list(map(lambda x, y: abs(x - y) < 1, true_rankings, predicted_rankings))) / len(true_rankings)
    # model.lamda *= 0.9
    print("training acc:{}".format(acc))


def validation(model, validation_gen):
    """
    validates a model which implements the ABSModelInterface class. the function output is the RMSE over the validation
     set.
    this function can and is used for single epoch validation round.
    :param model: the model to validate.
    :param validation_gen: a generator which iterates through all records to validate with. throws StopIteration when
     finished.
    :return float, RMSE over validation set. uses implemented RMSE function.
    """
    single_record = next(validation_gen)
    predicted_rankings = []
    true_rankings = []
    while single_record is not None:
        curr_user_id = single_record[1]
        curr_item_id = single_record[2]
        curr_rank = float(single_record[3])
        true_rankings.append(curr_rank)
        predicted_rankings.append(model.predict(user_id=curr_user_id, item_id=curr_item_id))
        try:
            single_record = next(validation_gen)
        except StopIteration:
            break

    return RMSE(true_ranks=true_rankings, predicted_ranks=predicted_rankings), accuracyEval(true_rankings,
                                                                                            predicted_rankings)


def train_base_model_grid_search(latent_features_size, train_data_path):
    # (lamda, gamma)
    # lamdas = np.arange(0.05, 0.1, 0.01)
    # gammas = np.arange(0.05, 0.1, 0.01)

    lamdas = [0.002]
    gammas = [0.05]

    params_to_test = list(zip(lamdas, gammas))
    model_preformences = []
    for idx, params in enumerate(params_to_test):
        model_preformences.append(
            TrainBaseModel(latent_features_size, train_data_path, lamda=params[0], gamma=params[1]))
    with open('grid_search_res', 'w') as gsr_file:
        for single_model_pref in model_preformences:
            gsr_file.write("rmse: {}, last_epoch: {}, Gamma: {}, Lambda: {}\n".format(single_model_pref[1],
                                                                                      single_model_pref[2],
                                                                                      single_model_pref[0].gamma,
                                                                                      single_model_pref[0].lamda))
    print('hello')


def TrainBaseModel(latent_features_size, train_data_path, max_ephocs=100, early_stopping=True, lamda=0.002, gamma=0.05):
    """
    Implement of the basic model described in Recommender Systems Handbook, chapter 3, section 3.3
    :param latent_features_size: number of features for the model.
    :param train_data_path: the path for the training data of the ranking, created via the load function.
    :return: trained model.
    """

    # print(latent_features_size, len(train_data_path))
    items_ids, users_ids, ranking_mean, ru_dict = get_unique_users_and_items(train_data_path)
    model = {}

    # split train_data into train and validation.
    train_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['train_split.csv'])
    valid_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['valid_split.csv'])
    test_data_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['userTestData.csv'])

    # split_and_save_train_validation(train_data_path,
    #                                 train_split_path=train_split_path,
    #                                 valid_split_path=valid_split_path,
    #                                 validation_percent=0.2)

    # randomly initialize U, b_u, b_i, p_u, q_i.
    model = BaseSVDModel(latent_features_size,
                         users_ids=users_ids,
                         items_ids=items_ids,
                         ranking_mean=ranking_mean,
                         lamda=lamda,
                         gamma=gamma)

    curr_rmse = float('inf')
    curr_epoch = 0

    while curr_epoch <= max_ephocs:
        # train the model over entire training set
        train_model(model, train_gen=load(train_data_path))
        # calculate RMSE over the validation, stop when is larger from prev iteration.
        temp_rmse, temp_rmsle = validation(model, validation_gen=load(test_data_path))
        print("Epoch #: {}, RMSE: {}, RMSLE: {}".format(curr_epoch, temp_rmse, temp_rmsle))
        if early_stopping and (curr_rmse - temp_rmse) < 0.000001:  # if negative the model is becoming worse
            break
        curr_rmse = temp_rmse
        curr_epoch += 1
    # todo: improve RMSE by updating γ and λ.(???)
    return model, curr_rmse, curr_epoch


def TrainImprovedModel(latent_features_size, train_data_path, max_ephocs=100, early_stopping=True):
    """
    Train an improved model from the prev SVD model, by the paper in https://dl.acm.org/doi/pdf/10.1145/1401890.1401944.
    """
    items_ids, users_ids, ranking_mean, ru_dict = get_unique_users_and_items(train_data_path)
    model = {}

    # split train_data into train and validation.
    train_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['train_split.csv'])
    valid_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['valid_split.csv'])
    test_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['userTestData.csv'])
    # split_and_save_train_validation(train_data_path,
    #                                 train_split_path=train_split_path,
    #                                 valid_split_path=valid_split_path,
    #                                 validation_percent=0.2)
    lamda = 0.007
    gamma1 = 0.005
    gamma2 = 0.015
    # randomly initialize U, b_u, b_i, p_u, q_i.
    model = SVDPlusModel(latent_features_size=latent_features_size,
                         users_ids=users_ids,
                         items_ids=items_ids,
                         ranking_mean=ranking_mean,
                         ru_dict=ru_dict,
                         lamda=lamda,
                         gamma1=gamma1,
                         gamma2=gamma2)

    curr_rmse = float('inf')
    curr_epoch = 0

    while curr_epoch <= max_ephocs:
        # train the model over entire training set
        train_model(model, train_gen=load(train_data_path))
        # calculate RMSE over the validation, stop when is larger from prev iteration.
        temp_rmse, temp_rmsle = validation(model, validation_gen=load(test_split_path))
        print("Epoch #: {}, RMSE: {}, RMSLE: {}".format(curr_epoch, temp_rmse, temp_rmsle))
        if early_stopping and (curr_rmse - temp_rmse) < 0.000001:  # if negative the model is becoming worse
            break
        curr_rmse = temp_rmse
        curr_epoch += 1
    # todo: improve RMSE by updating γ and λ.(???)
    return model, curr_rmse, curr_epoch


def TrainContentModel():
    pass


def TrainHybridModel():
    pass


if __name__ == '__main__':
    # train_data_path = "Data/userTrainDataSmall.csv"
    train_data_path = "Data/userTrainData.csv"

    # TrainImprovedModel(latent_features_size=3,
    #                    train_data_path=train_data_path,
    #                    max_ephocs=50,
    #                    early_stopping=True)

    TrainBaseModel(latent_features_size=3,
                   train_data_path=train_data_path,
                   max_ephocs=50,
                   early_stopping=True)

