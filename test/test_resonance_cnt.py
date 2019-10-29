import unittest
from fearquantlib.wavelib import *
from fearquantlib.config import QuantConfig as config


class TestResonance(unittest.TestCase):

    def test_fn(self):
        code = "SH.600478"
        df30 = get_df_of_code(code, "2019-07-04", "2019-08-01", KLType.K_30M)
        df15 = get_df_of_code(code, "2019-07-04", "2019-08-01", KLType.K_30M)

        df30 = __do_compute_df_bar(df30)
        df15 = __do_compute_df_bar(df15)

        cnt = resonance_cnt(df30, df15, "macd_bar")
        self.assertEqual(1, cnt)


if __name__ == '__main__':
    unittest.main()
