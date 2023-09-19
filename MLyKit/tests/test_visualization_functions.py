import unittest
from unittest.mock import patch
import pandas as pd
from MLyKit.data_visualization.plotting_functions import (
    plot_distribution,
    plot_correlation_matrix,
    plot_time_series,
)


class TestPlottingFunctions(unittest.TestCase):
    @patch("MLyKit.data_visualization.plotting_functions.plt.show")
    def test_plot_distribution(self, mock_show):
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45]})
        plot_distribution(df, "age")
        mock_show.assert_called_once()

    @patch("MLyKit.data_visualization.plotting_functions.plt.show")
    def test_plot_correlation_matrix(self, mock_show):
        df = pd.DataFrame({"age": [25, 30, 35], "score": [90, 85, 88]})
        plot_correlation_matrix(df)
        mock_show.assert_called_once()

    @patch("MLyKit.data_visualization.plotting_functions.plt.show")
    def test_plot_time_series(self, mock_show):
        df = pd.DataFrame({"time": [1, 2, 3], "value": [5, 6, 7]})
        plot_time_series(df, "time", "value")
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
