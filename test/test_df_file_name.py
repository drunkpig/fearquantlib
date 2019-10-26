import unittest
from fearquantlib.wavelib import KL_Period, df_file_name
from fearquantlib.config import QuantConfig as config


class TestDfFileName(unittest.TestCase):

    def test_file_name(self):
        model = config.DEV_MODEL
        ktype = KL_Period.KL_15
        code = "SH.12345"
        fname = df_file_name(code, ktype)
        self.assertEqual(fname, f'data/{model}{code}_{ktype}.csv')

        config.DEV_MODEL = "test_"
        fname  = df_file_name(code, ktype)
        self.assertTrue("test" in fname)


if __name__ == '__main__':
    unittest.main()
