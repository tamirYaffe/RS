import unittest
import RecommenderSystem


class TestRS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = "Data/userTrainDataSmall.csv"
        cls.train_data_gen = RecommenderSystem.load(file_path)
        print("setUpClass")

    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
