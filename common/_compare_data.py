from ._get_data import DataDict
from typing import List

import os
import shutil
import copydetect
import numpy as np

TMP_FOLDER = "_temp_files"
class CompareDict:
    # TODO: cellwise doesnt work yet --> get plagiarism scores is not supporting cellwise right now
    def __init__(self, data_dict: DataDict,
                 exclude_kw: str = None,
                 cellwise: bool = False):
        self.filetype = data_dict.filetype
        self.data = data_dict.data_dict
        self.all_frequency_values = data_dict.all_frequency_values

        self.markdown_scores,  self.code_scores = None, None

        self._exclude_kw = exclude_kw
        self._cellwise = cellwise
        self._mfl = [max(elements) for elements in zip(*self.all_frequency_values)]
        self._code_files = [] # file paths
        self._fingerprints = []

        if "ipynb" in self.filetype:
            self._notebook_routine()
        else:
            self._code_files = [f"{ddict['path']}/{filename}" for filename, ddict in self.data.items()]
            self._check_code()


    def _notebook_routine(self):
        self._notebooks_to_py_file()
        self._check_code()
        _del_temp_folder()
        pass

    def _check_code(self):
        self._get_fingerprints()

        file_names = list(self.data.keys())
        len_f = len(file_names)

        plag_scores = [None] * len_f
        iterator = list(range(len_f))
        for i in range(len_f-1):
            del iterator[0]
            scores = self._get_code_plagiarism_score(i, len_f, iterator)
            plag_scores[i] = scores
        plag_scores[-1] = [None]*len_f
        self.code_scores = np.asarray(plag_scores)

    def _get_code_plagiarism_score(self, candidate: int, len_f: int, iterator: List[int]) -> List[float]:
        cb1 = self._fingerprints[candidate]
        scores = [None] * len_f
        for j in iterator:
            cb2 = self._fingerprints[j]
            token_overlap, similarities, slices = copydetect.compare_files(cb1, cb2) # TODO: use token_overlap and slices
            scores[j] = sum(similarities) / len(similarities)
        return scores
    def _get_fingerprints(self):
        if self._cellwise:
            fingerprints = []
            for files in self._code_files:
                for file in files:
                    fp = copydetect.CodeFingerprint(file, 25, 1)
                    fingerprints.append(fp)
            self._fingerprints.append(fingerprints)
        else:
            for file in self._code_files:
                fp = copydetect.CodeFingerprint(file, 25, 1)
                self._fingerprints.append(fp)

    def _notebooks_to_py_file(self):
        os.makedirs(TMP_FOLDER, exist_ok=True)
        for file_name, file_dict in self.data.items():
            notebook_code = [cell for cell in file_dict["code_cells"] if self._exclude_kw not in cell] if self._exclude_kw else file_dict["code_cells"]
            program = "\n".join(notebook_code)
            path = TMP_FOLDER + f"/{file_name}.py"
            with open(path, "w", encoding="utf-8") as file:
                file.write(program)
            self._code_files.append(path)

def _del_temp_folder():
    shutil.rmtree(TMP_FOLDER)