"""
Pytest configuration file for Na-MRI denoising tests.

This file contains shared fixtures and configuration for pytest.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_data_4d():
    """Generate 4D test data for all tests."""
    from tests.test_implementations import generate_test_data
    return generate_test_data()

@pytest.fixture(scope="session")
def test_data_2d(test_data_4d):
    """Generate 2D test data for all tests."""
    return test_data_4d.reshape(-1, test_data_4d.shape[-1]) 