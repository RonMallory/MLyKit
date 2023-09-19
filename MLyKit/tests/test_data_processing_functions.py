import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from MLyKit.data_preprocessing.preprocessing_functions import (
    fill_string_col,
    fill_most_common,
    fill_average_round,
    downcast_dataframe,
    label_encode,
    separate_data,
    scale_features,
)


class TestDataPreprocessingFunctions(unittest.TestCase):
    def test_fill_string_col(self):
        df = pd.DataFrame({"name": [None, "Alice", "Bob"]})
        fill_string_col(df, ["name"], default="Unknown")
        expected_df = pd.DataFrame({"name": ["Unknown", "Alice", "Bob"]})
        assert_frame_equal(df, expected_df)

    def test_fill_average_round(self):
        df = pd.DataFrame({"score": [None, 50.0, 60.0]}, dtype=float)
        fill_average_round(df, ["score"])
        expected_df = pd.DataFrame({"score": [55.0, 50.0, 60.0]}, dtype=float)
        assert_frame_equal(df, expected_df)

    def test_fill_most_common(self):
        df = pd.DataFrame({"age": [None, 30, 30, 40]})
        df["age"] = df["age"].astype(
            pd.Int64Dtype()
        )  # Use Pandas nullable integer type
        fill_most_common(df, ["age"])
        expected_df = pd.DataFrame(
            {"age": [30, 30, 30, 40]}, dtype=pd.Int64Dtype()
        )
        assert_frame_equal(df, expected_df)

    def test_downcast_dataframe(self):
        df = pd.DataFrame({"count": [1, 2, 3], "value": [1.1, 2.2, 3.3]})
        downcast_dataframe(df)
        self.assertEqual(df["count"].dtype, "int8")
        self.assertEqual(df["value"].dtype, "float32")

    def test_label_encode(self):
        df = pd.DataFrame({"color": ["red", "green", "blue"]})
        df_encoded, label_encoders = label_encode(df, ["color"])
        self.assertEqual(df_encoded["color"].tolist(), [2, 1, 0])
        self.assertIn("color", label_encoders)

    def test_separate_data(self):
        df = pd.DataFrame(
            {"name": ["Alice", None, "Bob"], "age": [25, 30, None]}
        )
        df_no_missing, df_with_missing = separate_data(df)
        self.assertEqual(len(df_no_missing), 1)
        self.assertEqual(len(df_with_missing), 2)

    def test_scale_features(self):
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [5, 6, 7]})
        df_scaled, scaler = scale_features(df)

        self.assertTrue(
            np.allclose(df_scaled["feature1"].mean(), 0.0, atol=1e-7)
        )
        self.assertTrue(
            np.allclose(df_scaled["feature1"].std(ddof=0), 1.0, atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
