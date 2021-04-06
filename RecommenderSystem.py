import os
import time
from abc import ABC
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import math
import pickle

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
    :return: RMSE score
    """
    return mean_squared_error(true_ranks, predicted_ranks, squared=False)


def accuracyEval(true_ranks, predicted_ranks, threshold: float = 1.) -> float:
    """
    Calculate a flavor of accuracy - for each <prediction,true rank> pair, if
    the absolute difference between the pair is smaller than the given threshold,
    the prediction is considered a correct prediction, otherwise false.
    then, the classic accuracy score is calculated (T/T+F) and returned.
    :param true_ranks: actual ranking.
    :param predicted_ranks: the model predicted ranking.
    :param threshold: float
    :return: float, accuracy score
    """
    acc = sum(list(map(lambda x, y: abs(x - y) < threshold, true_ranks, predicted_ranks))) / len(true_ranks)
    return acc


def PredictRating(model, user_id, item_id):
    """
    Predict the rating for the input user and item
    :param model: the model that predict the rating
    :param user_id: the user id
    :param item_id: the item id
    :return: the predicted rating
    """
    return model.predict(user_id, item_id)


class ABSModelInterface(ABC):
    """
    Abstract class interface for all model to inherit.
    """
    def predict(self, user_id, item_id):
        """
        Predict the rating for the input user and item
        :param user_id: the user id
        :param item_id: the item id
        :return: the predicted rating
        """
        pass

    def correction(self, error, user_id, item_id):
        """
        Perform correction for svd models.
        :param error: error of predicted rank Vs true rank
        :param user_id: the user id
        :param item_id: the item id
        """
        pass


class BaseSVDModel(ABSModelInterface):
    """
    Base SVD model class.
    """
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

        if Bu == 0 and Bi == 0:
            pred_val = self.MU

        elif Bu == 0:
            pred_val = self.MU + Bi[0]

        elif Bi == 0:
            pred_val = self.MU + Bu[0]

        else:
            pred_val = self.MU + Bi[0] + Bu[0] + np.dot(user_latent_vec, item_latent_vec)

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


class SVDPlusModel(ABSModelInterface):
    """
    SVD plus model.
    """
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
        user_latent_vec = self.P.get(user_id, np.zeros(self.latent_features_size))
        item_latent_vec = self.Q.get(item_id, np.zeros(self.latent_features_size))

        Bi = self.BI.get(item_id, 0)
        Bu = self.BU.get(user_id, 0)
        Ru = self.RU.get(user_id, [])

        if Bu == 0 and Bi == 0:
            pred_val = self.MU

        elif Bu == 0:
            pred_val = self.MU + Bi[0]

        elif Bi == 0:
            pred_val = self.MU + Bu[0]

        else:
            y_i = np.array(list(map(lambda x: self.Y[x], self.RU[user_id]))).sum()
            # y_i = np.array([self.Y[it_id] for it_id in self.RU[user_id]]).sum()
            pred_val = self.MU + Bi[0] + Bu[0] + np.dot(item_latent_vec, (user_latent_vec + y_i / math.sqrt(len(Ru))))

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
    """
    Split training into train and validation sets.
    :param train_path: path to train file
    :param train_split_path: path to write new train file
    :param valid_split_path: path to write validation file
    :param validation_percent: percent of validation set of all training examples.
    """
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
        if valid_size <= (train_size + valid_size) * validation_percent:
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


def train_model(model, train_gen, learning_decay=0.97):
    """
    trains a model which implements the ABSModelInterface class.
    this function can and is used for single epoch training round.
    :param learning_decay: learning decay
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
    model.lamda *= learning_decay
    return acc


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
    print('started validation')

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

    print('finished validation')
    return RMSE(true_ranks=true_rankings, predicted_ranks=predicted_rankings), accuracyEval(true_rankings,
                                                                                            predicted_rankings)


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
    lamda = 0.005
    gamma = 0.05
    # split train_data into train and validation.
    train_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['train_split.csv'])
    valid_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['valid_split.csv'])
    # test_data_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['userTestData.csv'])

    split_and_save_train_validation(train_data_path,
                                    train_split_path=train_split_path,
                                    valid_split_path=valid_split_path,
                                    validation_percent=0.2)

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
        acc = train_model(model, train_gen=load(train_split_path))
        # calculate RMSE over the validation, stop when is larger from prev iteration.
        temp_rmse, temp_rmsle = validation(model, validation_gen=load(valid_split_path))

        log_result = "Epoch #: {}, RMSE: {}, train_ACC: {}, valid_ACC: {} \n".format(curr_epoch,
                                                                                     temp_rmse, acc, temp_rmsle)
        model_name = "svd_model_" + str(lamda)
        with open("Results/log_" + model_name + ".txt", 'a+') as log_file:
            log_file.write(log_result)
        # save model
        pickle.dump(model, open("Models/" + model_name + ".pickle", "wb"))

        print(log_result)
        if early_stopping and (curr_rmse - temp_rmse) < 0.000001:  # if negative the model is becoming worse
            break
        curr_rmse = temp_rmse
        curr_epoch += 1
    return model, curr_rmse, curr_epoch


def TrainImprovedModel(latent_features_size, train_data_path, max_ephocs=100, early_stopping=True, lamda=0.007,
                       gamma1=0.005, gamma2=0.015):
    """
    Train an improved model from the prev SVD model, by the paper in https://dl.acm.org/doi/pdf/10.1145/1401890.1401944.
    """
    items_ids, users_ids, ranking_mean, ru_dict = get_unique_users_and_items(train_data_path)

    # split train_data into train and validation.
    train_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['train_split.csv'])
    valid_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['valid_split.csv'])
    # test_split_path = os.sep.join(train_data_path.split(os.sep)[:-1] + ['userTestData.csv'])
    split_and_save_train_validation(train_data_path,
                                    train_split_path=train_split_path,
                                    valid_split_path=valid_split_path,
                                    validation_percent=0.2)

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
        acc = train_model(model, train_gen=load(train_split_path))
        # calculate RMSE over the validation, stop when is larger from prev iteration.
        temp_rmse, temp_rmsle = validation(model, validation_gen=load(valid_split_path))
        log_result = "Epoch #: {}, RMSE: {}, train_ACC: {}, valid_ACC: {} \n".format(curr_epoch,
                                                                                     temp_rmse, acc, temp_rmsle)
        model_name = "svd_plus_model_"+str(lamda)
        with open("Results/log_"+model_name+".txt", 'a+') as log_file:
            log_file.write(log_result)
        # save model
        pickle.dump(model, open("Models/"+model_name+".pickle", "wb"))

        print(log_result)
        if early_stopping and (curr_rmse - temp_rmse) < 0.000001:  # if negative the model is becoming worse
            break
        curr_rmse = temp_rmse
        curr_epoch += 1

    return model, curr_rmse, curr_epoch


class ContentModel(ABSModelInterface):
    """
    Content model class.
    """
    def __init__(self, items_hashmap, users_hashmap, model=RandomForestRegressor(n_estimators=10)):
        self.users_hashmap = users_hashmap
        self.items_hashmap = items_hashmap
        self.model = model
        self.trained = False

    def train(self, x_train, y_train):
        print('started training')
        self.model.fit(x_train, y_train)
        self.trained = True
        print('finished training')

    def predict(self, user_id, item_id):
        assert self.trained
        x_to_pred = self.users_hashmap[user_id] + self.items_hashmap[item_id]
        prediction = self.model.predict(np.array(x_to_pred).reshape(1, len(x_to_pred)))
        if prediction > 5:
            prediction = 5
        if prediction < 1:
            prediction = 1
        return prediction


def create_hashmaps(user_data_path, item_data_path):
    """
    Create and returns hash maps for user -> user features, and item -> item features.
    :param user_data_path: path to user data file.
    :param item_data_path: path to item data file.
    :return: the created hash maps.
    """
    le = preprocessing.LabelEncoder()
    scaler = preprocessing.MinMaxScaler()

    item_data_df = pd.read_csv(item_data_path)
    item_data_df.drop(axis=1,
                      labels=['name', 'neighborhood', 'address', 'postal_code', 'latitude', 'longitude', 'is_open'],
                      inplace=True)
    item_data_df['categories'] = [len(x.split(';')) for x in item_data_df['categories']]

    item_data_df['city'] = le.fit_transform(item_data_df['city'].astype(str))
    item_data_df['state'] = le.fit_transform(item_data_df['state'].astype(str))
    item_data_df['review_count'] = scaler.fit_transform(item_data_df['review_count'].values.reshape(-1, 1).astype(int))
    item_data_df['categories'] = scaler.fit_transform(item_data_df['categories'].values.reshape(-1, 1).astype(int))
    items_to_features_hash = item_data_df.set_index('business_id').T.to_dict('list')

    user_data_df = pd.read_csv(user_data_path)
    columns_to_drop = ['name', 'yelping_since', 'friends', 'elite', 'compliment_hot', 'compliment_more',
                       'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note',
                       'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer',
                       'compliment_photos']
    user_data_df.drop(axis=1, labels=columns_to_drop, inplace=True)
    user_data_df['review_count'] = scaler.fit_transform(user_data_df['review_count'].values.reshape(-1, 1).astype(int))
    user_data_df['useful'] = scaler.fit_transform(user_data_df['useful'].values.reshape(-1, 1).astype(int))
    user_data_df['funny'] = scaler.fit_transform(user_data_df['funny'].values.reshape(-1, 1).astype(int))
    user_data_df['cool'] = scaler.fit_transform(user_data_df['cool'].values.reshape(-1, 1).astype(int))
    user_data_df['fans'] = scaler.fit_transform(user_data_df['fans'].values.reshape(-1, 1).astype(int))
    user_data_df['average_stars'] = user_data_df['average_stars'].values.reshape(-1, 1).astype(float)
    users_to_features_hash = user_data_df.set_index('user_id').T.to_dict('list')

    return users_to_features_hash, items_to_features_hash


def pre_process_for_content_model(user_data_path, item_data_path, reviews_data_path, save_df=False):
    """
    Perform pre processing for the content model which create user and item hash maps and also create the X and Y sets
     for training.
    :param user_data_path: path to user data file.
    :param item_data_path: path to item data file.
    :param reviews_data_path: path to reviews file
    :param save_df: if true save the X and Y as a file.
    :return: the hash maps and training sets.
    """
    print('started preprocessing for content model')
    users_to_features_hash, items_to_features_hash = create_hashmaps(user_data_path, item_data_path)
    # Reviews - concat review to user id and item id
    X_Y_true = list()
    Y_true = list()
    review_gen = load(reviews_data_path)
    single_record = next(review_gen)
    while single_record is not None:
        curr_user_id = single_record[1]
        curr_item_id = single_record[2]
        curr_true_rank = float(single_record[3])
        curr_user_features = users_to_features_hash[curr_user_id]
        curr_item_features = items_to_features_hash[curr_item_id]
        X_Y_true.append(np.array(curr_user_features + curr_item_features + [curr_true_rank]))
        # Y_true.append(curr_true_rank)
        try:
            single_record = next(review_gen)
        except StopIteration as e:
            break
    X_Y_true = np.array(X_Y_true)
    # Y_true = np.array(Y_true)
    new_feature_df = pd.DataFrame(X_Y_true,
                                  columns=['if1', 'if2', 'if3', 'if4', 'if5', 'uf1', 'uf2', 'uf3', 'uf4', 'uf5', 'uf6',
                                           'Stars'])
    X = X_Y_true[:, :-1]
    Y = X_Y_true[:, -1]
    print('finished preprocessing for content model')
    if save_df:
        new_feature_df.to_csv('content_df.csv')
    return X, Y, users_to_features_hash, items_to_features_hash


def TrainContentModel(train_data_path, user_data_path, item_data_path):
    """
    Train and return the content model
    :param train_data_path: path to train file
    :param user_data_path: path to user data file.
    :param item_data_path: path to item data file.
    :return: the trained content model.
    """
    X, Y, users_to_features_hash, items_to_features_hash = pre_process_for_content_model(user_data_path=user_data_path,
                                                                                         item_data_path=item_data_path,
                                                                                         reviews_data_path=train_data_path)
    cm = ContentModel(items_hashmap=items_to_features_hash,
                      users_hashmap=users_to_features_hash)

    cm.train(x_train=X, y_train=Y)

    print(validation(cm, validation_gen=load('Data/userTestData.csv')))  # todo: remove before submission

    return cm


class HybridModel(ContentModel):
    """
    Hybrid model class.
    """
    def __init__(self, items_hashmap, users_hashmap, svd_model, model=RandomForestRegressor(n_estimators=10)):
        super().__init__(items_hashmap, users_hashmap, model)
        self.svd_model = svd_model

    def predict(self, user_id, item_id):
        assert self.trained
        svd_prediction = self.svd_model.predict(user_id, item_id)
        x_to_pred = self.users_hashmap[user_id] + self.items_hashmap[item_id] + [svd_prediction]
        prediction = self.model.predict(np.array(x_to_pred).reshape(1, len(x_to_pred)))
        if prediction > 5:
            prediction = 5
        if prediction < 1:
            prediction = 1
        return prediction


def pre_process_for_Hybrid_model(user_data_path, item_data_path, reviews_data_path, svd_model, save_df=False):
    """
    Perform pre processing for the hybrid model which create user and item hash maps and also create the X and Y sets
     for training.
    :param user_data_path: path to user data file.
    :param item_data_path: path to item data file.
    :param reviews_data_path: path to reviews file
    :param save_df: if true save the X and Y as a file.
    :param svd_model: a trained svd model.
    :return: the hash maps and training sets.
    """
    print('started preprocessing for hybrid model')
    users_to_features_hash, items_to_features_hash = create_hashmaps(user_data_path, item_data_path)
    # Reviews - concat review to user id and item id
    X_Y_true = list()
    review_gen = load(reviews_data_path)
    single_record = next(review_gen)
    while single_record is not None:
        curr_user_id = single_record[1]
        curr_item_id = single_record[2]
        curr_true_rank = float(single_record[3])
        curr_user_features = users_to_features_hash[curr_user_id]
        curr_item_features = items_to_features_hash[curr_item_id]
        curr_svd_predicted_rank = svd_model.predict(curr_user_id, curr_item_id)
        X_Y_true.append(np.array(curr_user_features + curr_item_features + [curr_svd_predicted_rank]
                                 + [curr_true_rank]))
        try:
            single_record = next(review_gen)
        except StopIteration as e:
            break

    X_Y_true = np.array(X_Y_true)
    new_feature_df = pd.DataFrame(X_Y_true,
                                  columns=['if1', 'if2', 'if3', 'if4', 'if5', 'uf1', 'uf2', 'uf3', 'uf4', 'uf5', 'uf6',
                                           'sf7', 'Stars'])
    X = X_Y_true[:, :-1]
    Y = X_Y_true[:, -1]
    print('finished preprocessing for hybrid model')
    if save_df:
        new_feature_df.to_csv('hybrid_df.csv')
    return X, Y, users_to_features_hash, items_to_features_hash


def TrainHybridModel(train_data_path, user_data_path, item_data_path, svd_model):
    """

    Train and return the hybrid model
    :param train_data_path: path to train file
    :param user_data_path: path to user data file.
    :param item_data_path: path to item data file.
    :param svd_model: a trained svd model.
    :return: the trained content model.
    """
    X, Y, users_to_features_hash, items_to_features_hash = pre_process_for_Hybrid_model(user_data_path=user_data_path,
                                                                                        item_data_path=item_data_path,
                                                                                        reviews_data_path=train_data_path,
                                                                                        svd_model=svd_model)

    hybrid_model = HybridModel(items_to_features_hash, users_to_features_hash, svd_model)

    hybrid_model.train(x_train=X, y_train=Y)

    print(validation(hybrid_model, validation_gen=load('Data/userTestData.csv')))  # todo: remove before submission

    return hybrid_model


def load_model(path_to_model: str) -> ABSModelInterface:
    """
    loads the pickled model stored in the given path parameter.
    :param path_to_model: str, absolute path to pickle file.
    return: ABSModelInterface class implementing object.
    """
    model = pickle.load(open(path_to_model, "rb"))
    return model


if __name__ == '__main__':
    start_time = time.time()
    # train_data_path = "Data/userTrainDataSmall.csv"
    train_data_path = "Data/userTrainData.csv"
    # model, curr_rmse, curr_epoch = TrainImprovedModel(latent_features_size=3,
    #                                                   train_data_path=train_data_path,
    #                                                   max_ephocs=30,
    #                                                   early_stopping=True)

    # TrainBaseModel(latent_features_size=3,
    #                train_data_path=train_data_path,
    #                max_ephocs=50,
    #                early_stopping=True)

    TrainContentModel(train_data_path=train_data_path, user_data_path='Data/yelp_user.csv', item_data_path='Data/yelp_business.csv')

    # model = load_model('Models/svd_model_0.005.pickle')
    # print(validation(model, validation_gen=load('Data/userTestData.csv')))

    # TrainHybridModel(train_data_path=train_data_path, user_data_path='Data/yelp_user.csv',
    #                  item_data_path='Data/yelp_business.csv', svd_model=model)

    print("computation time: %s minutes" % ((time.time() - start_time) / 60))
