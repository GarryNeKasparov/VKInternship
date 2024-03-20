import warnings
from typing import List

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def add_pca_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataframe with new pca-rotation feature.
    """
    new_coords = PCA().fit_transform(df[["lat", "lon"]])
    pca_lat = new_coords[:, 0]
    pca_lon = new_coords[:, 1]
    df["pca_lat"] = pca_lat
    df["pca_lon"] = pca_lon
    return df


class DBSCANCluster:
    """
    DBSCAN clustering with automatic eps computing.
    """

    def __init__(self, data, min_samples):
        eps = self.compute_eps(data=data, n_neighbors=min_samples)
        self.clf = DBSCAN(eps=eps, metric="haversine", min_samples=min_samples).fit(data)

    def compute_eps(self, data, n_neighbors):
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")
        neighbors_fit = neighbors.fit(data)
        distances, _ = neighbors_fit.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        kneedle = KneeLocator(
            range(1, len(distances) + 1),
            distances,
            S=1.0,
            curve="convex",
            direction="increasing",
        )
        return kneedle.knee_y

    def predict(self, X):
        return self.clf.labels_


def cluster_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataframe split into zones.
    """
    clstr = DBSCANCluster(df[["lat", "lon"]], min_samples=4)
    df["zone"] = clstr.predict(df[["lat", "lon"]])
    return df


def add_median_to_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataframe with computed median score for each zone.
    """
    assert "zone" in df.columns, '"Zone" feature is missing.'
    df["median_score"] = df[["zone", "score"]].groupby("zone").transform("median")
    return df


def find_k_closest_points(points_a: list, points_b: list, k: int) -> list:
    """
    Return array of k closest points from points_b
    for each point in points_a.
    """
    lat_long_a = np.array([(point[0], point[1]) for point in points_a])
    lat_long_b = np.array([(point[0], point[1]) for point in points_b])

    kdtree = KDTree(lat_long_b)

    _, indices = kdtree.query(lat_long_a, k=k)

    result = []

    for i in range(len(points_a)):
        closest_indices = indices[i]
        closest_values = np.hstack([points_b[idx][2:] for idx in closest_indices])
        result.append(closest_values)

    return result


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataset without useless features.
    """
    return df.drop(["id", "lat", "lon"], axis=1)


def get_featured_data(
    train_path: pd.DataFrame, test_path: pd.DataFrame, feat_path: pd.DataFrame, k: int
) -> List[pd.DataFrame]:
    """
    Return train/test dataframes with new features.
    train_path - path to train.csv
    test_path - path to test.csv
    feat_path - path to features.csv
    k - for finding closest points.
    """
    k = 2
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feat = pd.read_csv(feat_path)
    zones = cluster_data(pd.concat([train, test], axis=0))
    df = add_median_to_zones(zones)
    res = find_k_closest_points(df[["lat", "lon"]].values, feat.values, k)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df[[f"feat_{k}" for k in range(1, res[0].shape[0] + 1)]] = pd.DataFrame(
            res, index=df.index
        )

    train, test = df.iloc[: train.shape[0]], df.iloc[train.shape[0] :]
    test = select_features(test.drop(["score"], axis=1))
    train = select_features(train)
    return train, test
