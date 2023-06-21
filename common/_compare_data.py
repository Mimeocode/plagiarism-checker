from ._get_data import DataDict

TMP_FOLDER = "_temp_files"
class CompareDict:
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

        if "ipynb" in self.filetype:
            self._notebook_routine()
        else:
            self._check_code()


    def _notebook_routine(self):
        pass

    def _check_code(self):
        pass