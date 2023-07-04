import argparse
from common import DataDict, CompareDict

parser = argparse.ArgumentParser(prog="PlagiarismChecker",
                                 description="Runs the workflow for checking Plagiarism")

parser.add_argument("-f", "--filetype", type=str)
parser.add_argument("-a", "--archive", type=str)
parser.add_argument("-b", "--baseline", type=str, default=None)


def main(filetype: str, archive: str, baseline: str):
    dd = DataDict(archive=archive,
                  filetype=filetype,
                  baseline=baseline)
    dd.get_data()

    c = CompareDict(filetype=dd.filetype,
                    data_dict=dd.data_dict,
                    all_frequency_values=dd.all_frequency_values)
    c.run_comparison()



if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(filetype=args["filetype"],
         archive=args["archive"],
         baseline=args["baseline"])
