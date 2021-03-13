import unittest
import RecommenderSystem


class TestRS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")
        file_path = ""
        cls.train_data = RecommenderSystem.load(file_path)

    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
