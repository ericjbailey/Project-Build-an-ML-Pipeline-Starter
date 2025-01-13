import pandas as pd
import numpy as np
import scipy.stats
import pytest


def test_column_names(data: pd.DataFrame):
    """
    Test if the dataset has the expected column names in the correct order.
    """
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values
    assert list(expected_columns) == list(these_columns), "Column names do not match expected order."


def test_neighborhood_names(data: pd.DataFrame):
    """
    Test if the neighbourhood group names match the expected values.
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neigh = set(data['neighbourhood_group'].unique())
    assert set(known_names) == set(neigh), "Unexpected neighbourhood group names."


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC.
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)
    assert np.sum(~idx) == 0, "Some properties are outside the NYC boundaries."


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different from that of the reference dataset.
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()
    kl_divergence = scipy.stats.entropy(dist1, dist2, base=2)
    assert kl_divergence < kl_threshold, f"KL divergence {kl_divergence} exceeds threshold {kl_threshold}."


def test_row_count(data: pd.DataFrame, min_rows: int, max_rows: int):
    """
    Test if the number of rows in the dataset is within an acceptable range.
    """
    num_rows = data.shape[0]
    assert min_rows <= num_rows <= max_rows, f"Row count {num_rows} is outside the range {min_rows}-{max_rows}."


def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    Test if all prices in the dataset are within the specified range.
    """
    assert data['price'].between(min_price, max_price).all(), \
        f"Some prices are outside the range {min_price}-{max_price}."