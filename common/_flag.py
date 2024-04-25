from itertools import combinations
import pandas as pd
import numpy as np
from operator import add
from functools import reduce
from typing import Callable, Optional
from attrs import define, field, setters
from ._name_extractor import *


def _add_metric_distance(df: pd.DataFrame, metric_scaler: int = 10) -> pd.DataFrame:
    def _process_cell(cell: tuple):
        vals = filter(None, list(cell))
        if any(cell):
            # euclidean distance!
            scaled_vals = [(v * metric_scaler) ** 2 for v in
                           vals]  # scaling values since if 1 -> max distance is 1, if scaled distance increases when multiple metrics are max
            z = np.sqrt(reduce(add, scaled_vals)) / metric_scaler
            return cell + (z,)
        return None

    df = df.applymap(_process_cell)
    return df


@define(kw_only=True)
class Flagger:
    markdown_scores: list | None = field(on_setattr=setters.frozen)
    code_scores: list | None = field(on_setattr=setters.frozen)
    code_slices: list | None = field(on_setattr=setters.frozen)

    arguments: vars = field(on_setattr=setters.frozen)
    filetype: str = field(init=False)
    baseline: str | None = field(init=False)
    flagging_threshold: float = field(init=False)
    barren_threshold: float | None = field(init=False)

    file_names: list
    metric_cols: list = field(init=False)

    code_only: bool = field(on_setattr=setters.frozen)
    extractor: Optional[Callable] = field(init=False)
    flagging_df: pd.DataFrame = field(init=False)

    def _secondary_init(self):
        self.filetype = self.arguments["filetype"]
        self.baseline = self.arguments["baseline"]
        self.flagging_threshold = self.arguments["threshold"]
        self.barren_threshold = self.arguments["barrenthreshold"]

    def flag_submissions(self):
        self._secondary_init()

        self._extract_names()  # get names of submissions from file names
        self._flagging_df_from_score_lists()
        self.flagging_df.dropna(axis=0, how="any",
                                inplace=True)  # optional to get rid of incompletely analysed submissions

        if self.baseline and self.barren_threshold:
            self._remove_barren()
        self._flag_outliers(df=self.flagging_df, threshold=self.flagging_threshold)

    def _extract_names(self):
        self.extractor = {"None": None,
                          "canvas": canvas_rule,
                          "code_grade": code_grade_rule,
                          }[self.arguments["extract_name"]]

        self.file_names = [name.removesuffix(self.filetype) for name in self.file_names]
        if self.extractor:
            self.file_names = [self.extractor(name) if self.baseline not in name else name for name in self.file_names]

    def _flagging_df_from_score_lists(self):
        def __make_tuple(x, y):
            if isinstance(x, tuple):
                return x + (y,)
            return x, y

        dfs = []
        for lst in [self.markdown_scores, self.code_scores]:  # make dfs from not None score lists
            if lst is not None:
                df = pd.DataFrame(lst)
                # df.columns = df.index = self.file_names # not necessary since not accessed anywhere
                dfs.append(df)

        # combine dfs to one
        primary_df, *secondary_dfs = dfs
        for column in primary_df.columns:
            for sdf in secondary_dfs:
                primary_df[column] = primary_df[column].combine(sdf[column], lambda x, y: __make_tuple(x, y))
        primary_df = _add_metric_distance(primary_df)

        # make the relational df from previous square matrix df
        i, j = primary_df.shape
        assert i == j, f"ERROR, matrix expected to be square, got: {i}x{j}"
        data_size = int((i * j - j) / 2)  # number of unique relations between n elements
        self.metric_cols = [f"Metric_{column_index}" for column_index in range(len(dfs) + 1)]

        data = {'Submission 1': [None] * data_size,
                'Submission 2': [None] * data_size,
                'Code_Slices': [None] * data_size}
        for col in self.metric_cols:
            data[col] = [0] * data_size

        tmp_idx = 0  # this is faster than enumerate
        for ie, je in combinations(range(i), 2):  # populate data dir with submission pairs and scores
            data['Submission 1'][tmp_idx] = self.file_names[ie]
            data['Submission 2'][tmp_idx] = self.file_names[je]
            data['Code_Slices'][tmp_idx] = self.code_slices[ie][je]
            for k, col in enumerate(self.metric_cols):  # enumerate is used here since number of metrics usually < 10
                data[col][tmp_idx] = primary_df.iloc[ie, je][k]
            tmp_idx += 1

        self.flagging_df = pd.DataFrame(data).sort_values(self.metric_cols[-1], ascending=False)  # sorted by distance

    def _remove_barren(self) -> None:
        to_keep = [self.baseline]
        df = self.flagging_df

        # target_df is copy since flagging modifies underlying df
        target_df = df[df['Submission 1'].isin(to_keep) | df['Submission 2'].isin(to_keep)].copy()

        self._flag_outliers(df=target_df, threshold=self.barren_threshold, quantile=False)
        target_df_sus = target_df[target_df["Classification"] > 0]

        to_exclude = target_df_sus[["Submission 1", "Submission 2"]].values.tolist()
        to_exclude = set(sum(to_exclude, []))

        self.flagging_df = df[~df['Submission 1'].isin(to_exclude) | ~df['Submission 2'].isin(to_exclude)]

    def _flag_outliers(self, df: pd.DataFrame, threshold: float, quantile: bool = True) -> None:
        df["Classification"] = 0

        tmp_iter = self.metric_cols if not self.code_only else [self.metric_cols[1]]
        for ti in tmp_iter:
            q_threshold = df[ti].quantile(1 - threshold) if quantile else threshold
            df.loc[df[
                       ti] >= q_threshold, "Classification"] += 1  # if a metric is above the threshold increase classifcation of pairing

    def save_csv(self):
        self.flagging_df.to_csv("temp.csv")
