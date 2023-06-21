import os
import chardet
from json import load
from os import walk
from itertools import chain
from typing import Tuple
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import zipfile

class DataDict:
    def __init__(self, archive: str, filetype: str, baseline: str | None):
        self._detector = chardet.UniversalDetector()
        self._stopwords = set(stopwords.words('english'))
        self._vocabulary = []
        self._archive = archive

        self.filetype = filetype
        self.baseline = baseline
        self.path = None
        self.data_dict = {}
        self.all_frequency_values = []

        self._extract_archive()
        self._get_dict()

    def _extract_archive(self):
        self.path = self._archive.split(".zip")[0]
        with zipfile.ZipFile(self._archive, "r") as zip_ref:
            zip_ref.extractall(self.path)
        os.remove(self._archive)

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
            file_dict["markdown"] = []
            file_dict["word_frequencies"] = {}

            notebook_content = self._open_notebook(file_dict["path"] + file_name)
            if notebook_content is None:
                continue
            for cell in notebook_content["cells"]:
                ct = cell["cell_type"]
                s = "".join(cell["source"]) # this removes non utf-8 characters
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
        self.vocabulary = list(set(combined_markdown))

    def _get_word_frequencies(self):
        for _, file_dict in self.data_dict.items():
            _, tmp_freqs = self._extract_freq(file_dict)
            self._all_frequency_values.append(tmp_freqs)

    def _extract_freq(self, file_dict: dict) -> Tuple[dict, list]:
        file_dict["word_frequency"] = {}
        counts = Counter(sum(file_dict["markdown_cells"], []))
        tmp_freqs = []
        for vocab_word in self._vocabulary:
            file_dict["word_frequency"][vocab_word] = counts[vocab_word]
            tmp_freqs.append(counts[vocab_word])

            """
            OLD ONE: 
            if vocab_word in counts:
                file_dict["word_frequency"][vocab_word] = counts[vocab_word]
                tmp_freqs.append(counts[vocab_word])
            else:
                file_dict["word_frequency"][vocab_word] = 0
                tmp_freqs.append(0)
            """
        return file_dict, tmp_freqs

    def _open_notebook(self, path: str):
        try:
            self._detector.reset()
            for line in open(path, "rb"):
                self._detector.feed(line) # this is more robust in detecting encoding
                if self._detector.done: break
            self._detector.close()
            encoding = self._detector.result["encoding"]
            with open(path, encoding=encoding) as json_file:
                return load(json_file)
        except:
            print(f"{path} was problematic")
            pass
