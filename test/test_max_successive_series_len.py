import unittest
from fearquantlib.wavelib import __max_successive_series_len as max_len


class TestMxSuccSeriesLen(unittest.TestCase):

    def test_fn(self):
        arr = [1,7,3,4,5,2,4,5,6,1,0,4]
        l = max_len(arr)
        self.assertEqual(3,l)# 6,1,0


if __name__ == '__main__':
    unittest.main()
