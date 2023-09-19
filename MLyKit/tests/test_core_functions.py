import os
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from MLyKit.core.core_functions import (
    load_data,
    save_data,
    train_test_split,
)


class TestCoreFunctions(unittest.TestCase):
    def setUp(self):
        # Create sample data files if needed
        pass

    def test_load_data_csv(self):
        # Create a sample CSV file
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df.to_csv("test_data.csv", index=False)

        # Test load_data function for CSV
        loaded_df = load_data("test_data.csv", filetype="csv")
        assert_frame_equal(df, loaded_df)

    def test_load_data_excel(self):
        # Create a sample Excel file
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df.to_excel("test_data.xlsx", index=False)

        # Test load_data function for Excel
        loaded_df = load_data("test_data.xlsx", filetype="excel")
        assert_frame_equal(df, loaded_df)

    def test_save_data_csv(self):
        # Create a sample DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        # Test save_data function for CSV
        save_data(df, "saved_data.csv", filetype="csv")
        saved_df = pd.read_csv("saved_data.csv")
        assert_frame_equal(df, saved_df)

    def test_save_data_excel(self):
        # Create a sample DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        # Test save_data function for Excel
        save_data(df, "saved_data.xlsx", filetype="excel")
        saved_df = pd.read_excel("saved_data.xlsx")
        assert_frame_equal(df, saved_df)

    def test_train_test_split(self):
        # Create a sample DataFrame
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})

        # Test train_test_split function
        train, test = train_test_split(df, 0.5)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)

    def tearDown(self):
        # Remove any files created during the tests
        files_to_remove = [
            "saved_data.csv",
            "saved_data.xlsx",
            "test_data.csv",
            "test_data.xlsx",
        ]  # Add the names of files you wish to remove
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)


if __name__ == "__main__":
    unittest.main()
