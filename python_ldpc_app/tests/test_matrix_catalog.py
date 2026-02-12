"""Tests for matrix_catalog.py: filename parsing, rate lookup."""

import os
import pytest

from matrix_catalog import MatrixCatalog, MatrixInfo


class TestMatrixCatalog:
    def test_scan_database(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        assert len(catalog) > 0

    def test_bch_found(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        bch = [m for m in catalog.matrices if m.family == 'bch']
        assert len(bch) >= 1
        assert bch[0].n == 7
        assert bch[0].k == 4

    def test_wimax_rates(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        wimax = catalog.get_by_family('wimax')
        assert len(wimax) > 0
        rates = {m.rate for m in wimax}
        assert 0.5 in rates

    def test_get_by_rate_range(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        mid_rate = catalog.get_by_rate_range(0.45, 0.55)
        assert len(mid_rate) > 0
        for m in mid_rate:
            assert 0.45 <= m.rate <= 0.55

    def test_get_nearest_rate(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        nearest = catalog.get_nearest_rate(0.6, family='wimax')
        assert nearest is not None
        assert abs(nearest.rate - 0.6) < 0.2

    def test_get_lower_higher_rate(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        wimax = catalog.get_by_family('wimax')
        if len(wimax) < 2:
            pytest.skip("Not enough wimax matrices")

        # Pick a matrix in the middle
        mid = catalog.get_nearest_rate(0.66, family='wimax')
        if mid is None:
            pytest.skip("No wimax matrix near rate 0.66")

        lower = catalog.get_lower_rate(mid)
        higher = catalog.get_higher_rate(mid)

        if lower:
            assert lower.rate < mid.rate
        if higher:
            assert higher.rate > mid.rate

    def test_repr(self, matrix_catalog_path):
        if not os.path.isdir(matrix_catalog_path):
            pytest.skip("Channel_Codes_Database not found")
        catalog = MatrixCatalog(matrix_catalog_path)
        r = repr(catalog)
        assert 'MatrixCatalog(' in r
        assert 'matrices' in r
