import argparse
import pickle
from typing import Dict

import optuna
import pandas as pd
from catboost import CatBoostRegressor
from optuna.integration import CatBoostPruningCallback
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from features import get_featured_data


class Objective(object):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train, self.y_train, self.X_val, self.y_val = (
            X_train,
            y_train,
            X_val,
            y_val,
        )

    def __call__(self, trial):
        params = {
            "" "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.01, 0.1, log=True
            ),
            "eval_metric": "MAE",
        }
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        model = CatBoostRegressor(**params, verbose=0, random_seed=42)
        pruning_callback = CatBoostPruningCallback(trial, "MAE")
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=0,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
        )

        pruning_callback.check_pruned()
        pred = model.predict(self.X_val)
        return mean_absolute_error(self.y_val, pred)


def get_best_params(train: pd.DataFrame, n_trials: int, eval: bool = False) -> Dict:
    """
    Return best params for CatBoostRegressor using optuna optimization.
    train - train dataframe.
    n_trials - number of optuna's optimization trials.
    eval - indicates whether the final model should be evaluated on the test set.
    """
    X, y = train.drop(["score"], axis=1), train["score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0
    )
    sampler = TPESampler(seed=123)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(Objective(X_train, y_train, X_val, y_val), n_trials=n_trials)
    best_params = study.best_params
    if eval:
        pred = CatBoostRegressor(**best_params).fit(X_test)
        score = mean_absolute_error(y_test, pred)
        print(f"Test MAE: {score}")
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "-k",
        type=int,
        default=1,
        help="Number of feature points to add in data. \
                        See features.py/get_featured_data",
    )
    parser.add_argument(
        "-it", type=int, default=20, help="Number of optimization trials."
    )
    args = parser.parse_args()
    train, test = get_featured_data("train.csv", "test.csv", "features.csv", args.k)
    best_params = get_best_params(train, args.it, False)
    with open("best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    train.to_csv("feat_train.csv", index=False)
    test.to_csv("feat_test.csv", index=False)
