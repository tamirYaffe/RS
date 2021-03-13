
def load(filepath):
    """
    Load the csv data files into a structure of our choice dataframe or matrix.
    :param filepath: path for the csv file(should not be local).
    :return: data structure of the loaded file.
    """
    print(filepath)
    pass


def RMSE(true_ranks, predicted_ranks):
    """
    Calculate the root mean square error for the ranking.
    :param true_ranks: actual ranking.
    :param predicted_ranks: the model predicted ranking.
    :return:
    """
    print(true_ranks, predicted_ranks)
    pass


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


def TrainBaseModel(latent_features_size, train_data):
    """
    Implement of the basic model described in Recommender Systems Handbook, chapter 3, section 3.3
    :param latent_features_size: number of features for the model.
    :param train_data: the training data of the ranking.
    :return: trained model.
    """

    print(latent_features_size, len(train_data))
    model = {}
    # todo: split train_data into train and validation.
    # todo: randomly initialize U, b_u, b_i, p_u, q_i.
    # todo: initialize γ, λ with small values near 0.05.
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
