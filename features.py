import sys
from typing import List

import numpy as np
import pandas as pd


def generate_data(num_samples: int) -> pd.DataFrame:
    """
    Generates dataset with given number of samples.
    """
    assert num_samples > 362, "Number of samples must be greater than 362."
    ids = np.arange(num_samples)
    lat = np.random.uniform(55.57, 55.91, num_samples).astype(np.float32)
    lon = np.random.uniform(37.36, 37.84, num_samples).astype(np.float32)
    return pd.DataFrame({"id": ids, "lat": lat, "lon": lon})


def data_train_test_split(data: pd.DataFrame, test_size: float) -> List[pd.DataFrame]:
    """
    Splits whole dataset into train.csv and test.csv.
    """
    train_size = int((1 - test_size) * data.shape[0])
    score = np.random.randint(0, 100, train_size).astype(np.int32)
    train, test = data.iloc[:train_size].copy(), data.iloc[train_size]
    train["score"] = score
    return train, test


def generate_features(num_samples: int) -> pd.DataFrame:
    """
    Generates features.csv with given number of samples.
    """
    assert num_samples > 362, "Number of samples must be greater than 362."
    lat = np.random.uniform(55.57, 55.91, num_samples).astype(np.float32)
    lon = np.random.uniform(37.36, 37.84, num_samples).astype(np.float32)
    feat = np.arange(num_samples).astype(np.int32) % 363
    np.random.shuffle(feat)
    return pd.DataFrame({"lat": lat, "lon": lon, "0: 362": feat})


if __name__ == "__main__":
    np.random.seed(0)
    num_samples = int(sys.argv[1])
    feat_samples = int(sys.argv[2])
    df = generate_data(num_samples)
    train, test = data_train_test_split(df, 0.25)
    feat = generate_features(feat_samples)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    feat.to_csv("features.csv", index=False)
