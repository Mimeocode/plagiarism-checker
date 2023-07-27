from itertools import combinations
import pandas as pd
import numpy as np
from operator import sub, add
from functools import reduce
from typing import Callable
from attrs import define, field, Factory, validators, setters


def _add_metric_distance(df: pd.DataFrame, metric_scaler: int = 10) -> pd.DataFrame:
    def _process_cell(cell: tuple):
        vals = filter(None, list(cell))
        if any(cell):
            scaled_vals = [v * metric_scaler for v in vals] # scaling values since if 1 -> max distance is 1, if scaled distance increases when multiple metrics are max
            z = (np.abs(reduce(add, scaled_vals))+np.abs(reduce(sub, scaled_vals))) / len(cell) / metric_scaler
            return *vals, z
        return None
    df = df.applymap(_process_cell)
    return df


def _canvas_rule(x):
    lst = x.split("_")
    try:
        if lst[1] == "LATE":
            return "_".join([lst[1],lst[-1][:7]])
        return lst[-1][:7]
    except:
        return x


@define(kw_only=True)
class Flagger:
    markdown_scores: list | None = field(on_setattr=setters.frozen)
    code_scores: list | None = field(on_setattr=setters.frozen)

    arguments: vars = field(on_setattr=setters.frozen)
    filetype: str = field(init=False)
    baseline: str | None = field(init=False)
    flagging_threshold: float = field(init=False)
    barren_threshold: float | None = field(init=False)

    file_names: list
    metric_cols: list = field(init=False)

    code_only: bool = field(on_setattr=setters.frozen)
    extractor: Callable | None = field(default=_canvas_rule, on_setattr=setters.frozen)
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
        self.flagging_df.dropna(axis=0, how="any", inplace=True)  # optional to get rid of incompletely analysed submissions

        if self.baseline and self.barren_threshold:
            self._remove_barren()
        self._flag_outliers(df=self.flagging_df, threshold=self.flagging_threshold)

    def _extract_names(self):
        self.file_names = [name.removesuffix(self.filetype) for name in self.file_names]
        if self.extractor is not None:
            self.file_names = [self.extractor(name) for name in self.file_names]

    def _flagging_df_from_score_lists(self):
        def __make_tuple(x, y: list):
            y.insert(0, x)
            return tuple(y)

        dfs = []
        for lst in [self.markdown_scores, self.code_scores]:  # make dfs from not None score lists
            if lst is not None:
                df = pd.DataFrame(lst)
                # df.columns = df.index = self.file_names # not necessary since not accessed anywhere
                dfs.append(df)

        # combine dfs to one
        primary_df, *secondary_dfs = dfs
        for column in primary_df.columns:
            # TODO: combining multiple not working  yet
            primary_df[column] = primary_df[column].combine([secondary_dfs], lambda x, y: __make_tuple(x, [ydf[column] for ydf in y]))
        primary_df = _add_metric_distance(primary_df)

        # make the relational df from previous square matrix df
        i, j = primary_df.size
        assert i == j, f"ERROR, matrix expected to be square, got: {i}x{j}"
        data_size = int((i * j - j) / 2)  # number of unique relations between n elements
        self.metric_cols = [f"Metric_{column_index}" for column_index in range(len(dfs) + 1)]

        data = {'Submission 1': [None] * data_size,
                'Submission 2': [None] * data_size}
        for col in self.metric_cols:
            data[col] = [0] * data_size

        tmp_idx = 0  # this is faster than enumerate
        for ie, je in combinations(range(i), 2):  # populate data dir with submission pairs and scores
            data['Submission 1'][tmp_idx] = self.file_names[ie]
            data['Submission 2'][tmp_idx] = self.file_names[je]
            for k, col in enumerate(self.metric_cols):  # enumerate is used here since number of metrics usually < 10
                data[col][tmp_idx] = primary_df.iloc[ie, je][k]
            tmp_idx += 1

        self.flagging_df = pd.DataFrame(data).sort_values("combined", ascending=False)

    def _remove_barren(self):
        to_keep = [self.baseline]
        df = self.flagging_df

        # target_df is copy since flagging modyfies underlying df
        target_df = df[df['Submission 1'].isin(to_keep) | df['Submission 2'].isin(to_keep)].copy()
        self._flag_outliers(df=target_df, threshold=self.barren_threshold)
        target_df_sus = target_df[target_df["Classification"] > 0]

        to_exclude = target_df_sus[["Submission 1", "Submission 2"]].tolist()  # THIS COULD BE AN ISSUE
        to_exclude = [str(x) for x in set(to_exclude)]

        self.flagging_df = df[~df['Submission 1'].isin(to_exclude) | ~df['Submission 2'].isin(to_exclude)]

    def _flag_outliers(self, df, threshold):
        df["Classification"] = 0

        tmp_iter = self.metric_cols if not self.code_only else [self.metric_cols[0]]
        for ti in tmp_iter:
            threshold = df[ti].quantile(1 - threshold)
            df.loc[df[ti] >= threshold, "Classification"] += 1  # if a metric is above the threshold increase classifcation of pairing

    def save_csv(self):
        self.flagging_df.to_csv("temp.csv")
