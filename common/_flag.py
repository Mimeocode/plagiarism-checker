from itertools import combinations
import pandas as pd
import numpy as np
from operator import sub, add
from functools import reduce
from typing import Callable
from attrs import define, field, Factory, validators


def _add_distance(df: pd.DataFrame) -> pd.DataFrame:
    def _process_cell(cell: tuple):
        vals = filter(None, list(cell))
        if any(cell):
            z = (np.abs(reduce(add, vals))+np.abs(reduce(sub, vals)))/len(cell)
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


@define(kw_only = True)
class Flagger:
    markdown_scores: list | None
    code_scores: list | None
    filetype: str
    file_names: list

    code_only: bool
    extractor: Callable | None = field(default=_canvas_rule)
    flagging_df: pd.DataFrame = field(init=False)

    def flag_sumbissions(self):
        self._extract_names()  # get names of submissions from file names
        self._flagging_df_from_score_lists()
        self.flagging_df.dropna(axis=0,how="any",inplace=True)  # optional to get rid of incompletely analysed submissions

    def _extract_names(self):
        self.file_names = [name.removesuffix(self.filetype) for name in self.file_names]
        if self.extractor is not None:
            self.file_names = [self.extractor(name) for name in self.file_names]

    def _flagging_df_from_score_lists(self):
        dfs = []
        for l in [self.markdown_scores, self.code_scores]:  # make dfs from not None score lists
            if l is not None:
                df = pd.DataFrame(l)
                # df.columns = df.index = self.file_names # not necessary since not accessed anywhere
                dfs.append(df)

        # combine dfs to one
        primary_df, *secondary_dfs = dfs
        for column in primary_df.columns:
            primary_df[column] = primary_df[column].combine(secondary_dfs, lambda x: tuple([xdf[column] for xdf in x]))
        primary_df = _add_distance(primary_df)

        # make the relational df from previous square matrix df
        i, j = primary_df.size
        assert i == j, f"ERROR, matrix expected to be square, got: {i}x{j}"
        data_size = int((i * j - j) / 2)  # number of unique relations between n elements
        metric_cols = [f"Metric_{column_index}" for column_index in range(len(dfs) + 1)]

        data = {'Submission 1': [None] * data_size,
                'Submission 2': [None] * data_size}
        for col in metric_cols:
            data[col] = [0] * data_size

        tmp_idx = 0  # this is faster than enumerate
        for ie, je in combinations(range(i), 2):  # populate data dir with submission pairs and scores
            data['Submission 1'][tmp_idx] = self.file_names[ie]
            data['Submission 2'][tmp_idx] = self.file_names[je]
            for k, col in enumerate(metric_cols):  # enumerate is used here since number of metrics usually < 10
                data[col][tmp_idx] = primary_df.iloc[ie, je][k]
            tmp_idx += 1

        self.flagging_df = pd.DataFrame(data).sort_values("combined", ascending=False)

    #TODO: flagging using threshold etc.