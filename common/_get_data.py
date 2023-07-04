import os
import chardet
from json import load
from os import walk
from itertools import chain
import shutil
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import zipfile
from attrs import validators, setters, define, field


@define(kw_only=True)
class DataDict:
    _detector = chardet.UniversalDetector()
    _stopwords = set(stopwords.words('english'))
    _vocabulary: list = []

    archive: str = field(validator=validators.instance_of(str), on_setattr=setters.frozen)
    filetype: str = field(validator=validators.instance_of(str), on_setattr=setters.frozen)
    baseline: str | None

    path: str = ""
    data_dict: dict = {}
    all_frequency_values: list = []

    def get_data(self):
        self._extract_archive()
        self._get_dict()
        self._del_tmp_archive_folder()  #-> removes folder of submissions

    def _extract_archive(self):
        self.path = self.archive.split(".zip")[0]
        with zipfile.ZipFile(self.archive, "r") as zip_ref:
            zip_ref.extractall(self.path)
        os.remove(self.archive)

    def _get_dict(self):
        for _, _, filenames in walk(self.path):
            for filename in filenames:
                if filename.endswith(self.filetype):
                    self.data_dict[filename] = {}
                    self.data_dict[filename]["path"] = self.path

        if "ipynb" in self.filetype:
            self._notebook_extractor()
            self._get_word_frequencies()

    def _notebook_extractor(self):
        lemmatizer = WordNetLemmatizer()

        combined_markdown = {}
        for file_name, file_dict in self.data_dict.items():
            file_dict["markdown_cells"] = []
            file_dict["code_cells"] = []
            file_dict["word_frequencies"] = {}

            notebook_content = self._open_notebook(f"{file_dict['path']}/{file_name}")
            if notebook_content is None:
                continue
            for cell in notebook_content["cells"]:
                ct = cell["cell_type"]
                s = "".join(cell["source"])  # this removes non utf-8 characters
                if ct == "code":
                    file_dict["code_cells"].append(s)
                elif ct == "markdown":
                    file_dict["markdown_cells"].append(word_tokenize(s))

            for word in chain.from_iterable(file_dict["markdown_cells"]):
                if word not in self._stopwords:
                    word = lemmatizer.lemmatize(word.lower())
                    if word not in combined_markdown:
                        combined_markdown[word] = 1
                    else:
                        combined_markdown[word] += 1
                    if self.baseline and file_name == self.baseline:
                        combined_markdown[word] -= 2
        self._vocabulary = list(set(combined_markdown))

    def _get_word_frequencies(self):
        [self._extract_freq(file_dict) for _, file_dict in self.data_dict.items()]

    def _extract_freq(self, file_dict: dict):
        file_dict["word_frequency"] = {}
        counts = Counter(sum(file_dict["markdown_cells"], []))
        for vocab_word in self._vocabulary:
            file_dict["word_frequency"][vocab_word] = counts[vocab_word]
        self.all_frequency_values.append(list(file_dict["word_frequency"].values()))

    def _open_notebook(self, path: str):
        try:
            self._detector.reset()
            for line in open(path, "rb"):
                self._detector.feed(line)  # this is more robust in detecting encoding

                if self._detector.done:
                    break
            self._detector.close()
            encoding = self._detector.result["encoding"]
            with open(path, encoding=encoding) as json_file:
                return load(json_file)
        except:
            print(f"{path} was problematic")
            pass

    def _del_tmp_archive_folder(self):
        shutil.rmtree(self.archive.split(".zip")[0])
