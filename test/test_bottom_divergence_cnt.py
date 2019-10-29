import unittest

from fearquantlib.wavelib import *


class TestBottomDivergenceCnt(unittest.TestCase):

    def test_fun(self):
        code = "SH.600703"
        df = get_df_of_code(code, "2019-09-20", "2019-10-21", KLType.K_30M)
        df15 = __do_compute_df_bar(df)
        ct0 = bottom_divergence_cnt(df15, "macd_bar", "close")
        self.assertEqual(0, ct0)

        ct2 = bottom_divergence_cnt(df15[:-4], "macd_bar", "close")
        self.assertEqual(2, ct2)


if __name__ == '__main__':
    unittest.main()
