import unittest

from fearquantlib.wavelib import *


class TestBarGreenWaveCnt(unittest.TestCase):

    def test_fun(self):
        code = "SH.600703"
        df = get_df_of_code(code, "2019-09-20", "2019-10-21", KLType.K_30M)
        df15 = do_compute_df_bar(df)

        ct_4 = bar_green_wave_cnt(df15[:-4])
        self.assertEqual(4, ct_4)

        ct_0 = bar_green_wave_cnt(df15)
        self.assertEqual(0, ct_0)



if __name__ == '__main__':
    unittest.main()