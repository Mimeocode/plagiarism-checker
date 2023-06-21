import argparse
from common import DataDict, CompareDict

parser = argparse.ArgumentParser(prog="PlagiarismChecker",
                                 description="Runs the workflow for checking Plagiarism")

parser.add_argument("-f", "--filetype", type=str)
parser.add_argument("-a", "--archive", type=str)
parser.add_argument("-b", "--baseline", type=str, default=None)


def main(filetype: str, archive: str, baseline: str):
    dd = DataDict(archive, filetype, baseline)
    c = CompareDict(data_dict=dd)



if __name__ == '__main__':
    main(filetype=parser["filetype"],
         archive=parser["archive"],
         baseline=parser["baseline"])
