import unittest
from fearquantlib.wavelib import *
from fearquantlib.wavelib import __do_compute_df_bar as do_compute_df_bar
from fearquantlib.config import QuantConfig as config


class TestResonance(unittest.TestCase):

    def test_fn(self):
        code = "SH.600478"
        df30 = get_df_of_code(code, "2019-07-04", "2019-08-01", KLType.K_30M)
        df15 = get_df_of_code(code, "2019-07-04", "2019-08-01", KLType.K_30M)

        df30 = do_compute_df_bar(df30, "KL_30")
        df15 = do_compute_df_bar(df15, "KL_15")

        cnt = resonance_cnt(df30, df15, "macd_bar")
        self.assertEqual(1, cnt)


if __name__ == '__main__':
    unittest.main()
