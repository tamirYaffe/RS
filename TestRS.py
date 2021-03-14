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

    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
