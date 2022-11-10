from pathlib import Path
from statistics import mean, stdev
import pandas as pd
import csv

EXPERIMENT_FOLDER = Path(
    __file__).parent.parent.resolve() / "data" / "experiment"

DATA_FOLDER = Path(
    __file__).parent.parent.resolve() / "data"


def load_data():
    test_data = []
    experiment_data = []
    for f in EXPERIMENT_FOLDER.iterdir():
        if f.stem.endswith("_test"):
            df = pd.read_csv(f, header=0, index_col=0)
            df["subject"] = f.stem.replace("_test", "")
            test_data.append(df)
        else:
            df = pd.read_csv(f, header=0, index_col=0)
            df["subject"] = f.stem
            experiment_data.append(df)

    test_df = pd.concat(test_data, ignore_index=True)
    experiment_df = pd.concat(experiment_data, ignore_index=True)

    test_df = test_df.astype(
        {"answer": "string", "question": "string", "subject": "category"})
    experiment_df = experiment_df.astype(
        {"answer": "string", "question": "string", "subject": "category"})

    return test_df, experiment_df


def get_correct_scores(df: pd.DataFrame) -> dict:

    scores = dict()
    scores["count"] = df.groupby(
        "subject")["correct"].value_counts().xs(True, level=1)
    scores["frequency"] = df.groupby("subject")["correct"].value_counts(
        normalize=True).xs(True, level=1)
    scores["mean"] = df.groupby("subject")["correct"].value_counts(
        normalize=True).xs(True, level=1).mean()
    scores["std"] = df.groupby("subject")["correct"].value_counts(
        normalize=True).xs(True, level=1).std()

    return scores


def get_trial_stats(df: pd.DataFrame) -> dict:

    stats = dict()

    stats["count"] = df["subject"].value_counts(sort=False)
    stats["mean"] = df["subject"].value_counts(sort=False).mean()
    stats["std"] = df["subject"].value_counts(sort=False).std()

    return stats


def get_flipped(df: pd.DataFrame, is_test: bool = False, reverse: bool = False) -> pd.DataFrame:
    if reverse:
        if is_test:
            swahili = []
            with open(DATA_FOLDER / "swahili.csv", encoding="utf-8") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    swahili.append(row[2])

            return df[~df.isin({"question": swahili}).any(1)]
        else:
            df[df["flipped"] == False]
    else:
        # unfortunately we forgot to save the flipping information for the tests s we need to infer from the word list
        if is_test:
            swahili = []
            with open(DATA_FOLDER / "swahili.csv", encoding="utf-8") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    swahili.append(row[2])

            return df[df.isin({"question": swahili}).any(1)]
        else:
            return df[df["flipped"] == True]


def get_flipping_stats(df: pd.DataFrame, is_test: bool = False) -> dict:

    stats = dict()

    flipped_df = get_flipped(df, is_test)

    df_grouped = df.groupby("subject", observed=True)

    flipped_grouped = flipped_df.groupby("subject", observed=True)

    stats["count"] = flipped_df["subject"].value_counts(sort=False)

    stats["frequency"] = []
    stats["freq_keys"] = []

    for key, group in flipped_grouped.groups.items():
        stats["frequency"].append(len(group)/len(df_grouped.get_group(key)))
        stats["freq_keys"].append(key)

    stats["mean"] = mean(stats["frequency"])
    stats["std"] = stdev(stats["frequency"])

    return stats


if __name__ == "__main__":

    test_df, experiment_df = load_data()

    df = get_flipped(test_df, is_test=True, reverse=True)

    print(df.info())
