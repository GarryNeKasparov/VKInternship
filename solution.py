import argparse
import os
import pickle

import pandas as pd
from catboost import CatBoostRegressor

from features import get_featured_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making submission")
    parser.add_argument(
        "--params", type=str, default=None, help="Path to pretrained model dict"
    )
    parser.add_argument("--train", type=str, default=None, help="Path to featured train")
    parser.add_argument("--test", type=str, default=None, help="Path to featured test")
    parser.add_argument(
        "-k",
        type=int,
        default=2,
        help="Number of feature points to add in data. \
                        See features.py/get_featured_data",
    )
    args = parser.parse_args()

    if args.params and os.path.exists(args.params):
        with open(args.params, "rb") as f:
            params = pickle.load(f)
        model = CatBoostRegressor(**params)
    else:
        print("Using default parameters.")
        model = CatBoostRegressor()
    if (
        args.test
        and os.path.exists(args.test)
        and args.train
        and os.path.exists(args.train)
    ):
        test = pd.read_csv(args.test)
        train = pd.read_csv(args.train)
    else:
        train, test = get_featured_data("train.csv", "test.csv", "features.csv", k=args.k)
    model.fit(train.drop(["score"], axis=1), train["score"])
    pred = model.predict(test)
    sample = pd.read_csv("submission_sample.csv")
    sample["score"] = pred
    sample.to_csv("submission.csv", index=False)
