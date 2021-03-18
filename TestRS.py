import unittest
import RecommenderSystem
import numpy as np


class TestRS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = "Data/userTrainDataSmall.csv"
        cls.train_data_gen = RecommenderSystem.load(file_path)
        print("setUpClass")

    def test_rmse(self):
        a1 = np.array([2, 2])
        a2 = np.array([3, 3])
        self.assertEqual(RecommenderSystem.RMSE(a1, a2), 1)

    def test_get_user_and_items_cnts(self):
        file_path = "Data/userTrainData.csv"
        items_ids, users_ids, ranking_mean, ru_dict = RecommenderSystem.get_unique_users_and_items(file_path)
        print('ranking_mean = {}'.format(ranking_mean))
        self.assertEqual(len(items_ids), 159168)
        self.assertEqual(len(users_ids), 239007)

    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
