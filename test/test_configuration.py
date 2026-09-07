"""Tests for optional display configuration."""

import pandas as pd

from qmm import configure_pandas_display


def test_configure_pandas_display_applies_requested_options():
    with pd.option_context('display.max_columns', 20, 'display.max_rows', 60,
                           'display.max_colwidth', 50, 'display.width', 80,
                           'display.html.use_mathjax', False):
        configure_pandas_display(max_columns=8, max_rows=12, max_colwidth=40,
                                 display_width=100)
        assert pd.get_option('display.max_columns') == 8
        assert pd.get_option('display.max_rows') == 12
        assert pd.get_option('display.max_colwidth') == 40
        assert pd.get_option('display.width') == 100
        assert pd.get_option('display.html.use_mathjax') is True
