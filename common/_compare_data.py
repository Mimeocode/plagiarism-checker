import os
import uuid
import shutil
import copydetect
from copydetect.utils import highlight_overlap
import numpy as np
from numpy.linalg import norm
from numpy import dot
from attrs import define, field, validators, setters


@define(kw_only=True)
class CompareDict:
    """Comparison Object that allows for comparison of files."""
    exclude_kw: str | None = field(validator=validators.optional(validators.instance_of(str)), default=None, on_setattr=setters.frozen)
    cellwise: bool | None = field(validator=validators.optional(validators.instance_of(bool)), default=None, on_setattr=setters.frozen)

    arguments: vars = field(on_setattr=setters.frozen)
    filetype: str = field(init=False)
    data_dict: dict = field(validator=validators.instance_of(dict))
    all_frequency_values: list = field(validator=validators.instance_of(list))

    markdown_scores = field(init=False)  # list | ndarray | None
    code_scores = field(init=False)  # list | ndarray | None
    code_slices = field(init=False)  # list | ndarray | None

    _mfl: list = field(init=False)
    _code_files: list = []
    _fingerprints: list = []
    _TMP_FOLDER: str = f"_temp_files_{uuid.uuid4()}"

    # TODO: cellwise doesnt work yet --> get plagiarism scores is not supporting cellwise right now

    def _secondary_init(self):
        self.filetype = self.arguments["filetype"]

    def run_comparison(self):
        self._secondary_init()

        self._mfl = [max(elements) for elements in zip(*self.all_frequency_values)]
        if "ipynb" in self.filetype:
            self._notebooks_to_py_files()
            self._check_code()
            self._del_tmp_folder()
            self._check_markdown()
        else:
            self._code_files = [f"{file_dict['path']}/{filename}" for filename, file_dict in self.data_dict.items()]
            self._check_code()

    def _check_markdown(self):
        len_d = len(self.data_dict)

        plag_scores = [None]*len_d
        iterator = list(range(len_d))
        for i in range(len_d-1):
            del iterator[0]
            plag_scores[i] = self._get_markdown_plagiarism_score(i, iterator)
        plag_scores[-1] = [None]*len_d
        self.markdown_scores = np.asarray(plag_scores)

    def _check_code(self):
        self._generate_fingerprints()
        file_names = list(self.data_dict.keys())
        len_f = len(file_names)

        plag_scores = [None] * len_f
        plag_slices = [None] * len_f
        iterator = list(range(len_f))
        for i in range(len_f-1):
            del iterator[0]
            plag_scores[i], plag_slices[i] = self._get_code_plagiarism_score(i, len_f, iterator)
        plag_scores[-1], plag_slices[-1] = [None]*len_f, [(None, None)]*len_f
        self.code_scores, self.code_slices = np.asarray(plag_scores), np.asarray(plag_slices)

    def _get_code_plagiarism_score(self,
                                   candidate: int,
                                   len_f: int,
                                   iterator: list[int]) -> tuple[list[float|None], list[tuple[str, str]|tuple[None, None]]]:
        cb1 = self._fingerprints[candidate]
        scores, slices = [None] * len_f, [(None, None)] * len_f
        for j in iterator:
            cb2 = self._fingerprints[j]
            _, similarities, slice = copydetect.compare_files(cb1, cb2)  # TODO: use token_overlap and slices
            curr_score = sum(similarities) / len(similarities)
            scores[j] = curr_score

            code1, _ = highlight_overlap(cb1.raw_code, slice[0], ">>", "<<")
            code2, _ = highlight_overlap(cb2.raw_code, slice[1], ">>", "<<")
            slices[j] = (code1, code2)

        return scores, slices

    def _get_markdown_plagiarism_score(self, cand: int, iterator: list[int]) -> list[float]:
        def _cos_similarity(elem1: list[int], elem2: list[int]) -> float:
            a = np.array([x/y if y > 0 else 0 for x, y in zip(elem1, self._mfl)])
            b = np.array([x/y if y > 0 else 0 for x, y in zip(elem2, self._mfl)])
            score = float(dot(b.T, a) / (norm(a) * norm(b)))
            return score

        len_f = len(self.all_frequency_values)
        nb1 = self.all_frequency_values[cand]
        scores = [None] * len_f
        for i in iterator:
            nb2 = self.all_frequency_values[i]
            scores[i] = _cos_similarity(nb1, nb2)
        return scores

    def _generate_fingerprints(self):
        if self.cellwise:
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

    def _notebooks_to_py_files(self):
        os.makedirs(self._TMP_FOLDER, exist_ok=True)
        for file_name, file_dict in self.data_dict.items():
            notebook_code = [cell for cell in file_dict["code_cells"] if self.exclude_kw not in cell] if self.exclude_kw else file_dict["code_cells"]
            program = "\n".join(notebook_code)
            path = self._TMP_FOLDER + f"/{file_name}.py"
            with open(path, "w", encoding="utf-8") as file:
                file.write(program)
            self._code_files.append(path)

    def _del_tmp_folder(self):
        shutil.rmtree(self._TMP_FOLDER)
