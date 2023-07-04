import argparse
from common import *

parser = argparse.ArgumentParser(prog="PlagiarismChecker",
                                 description="Runs the workflow for checking Plagiarism")

parser.add_argument("-f", "--filetype", type=str)
parser.add_argument("-a", "--archive", type=str)
parser.add_argument("-b", "--baseline", type=str, default=None)
parser.add_argument("-c", "--codeonly", type=bool, default=True)


def main(filetype: str, archive: str, baseline: str, code_only: bool):
    dd = DataDict(filetype=filetype,
                  archive=archive,
                  baseline=baseline)
    dd.get_data()

    c = CompareDict(filetype=filetype,
                    data_dict=dd.data_dict,
                    all_frequency_values=dd.all_frequency_values)
    c.run_comparison()

    code_only = True if "ipynb" not in filetype else code_only  # dont allow markdown comparison for non-notebook files
    f = Flagger(filetype=filetype,
                file_names=list(dd.data_dict.keys()),
                code_only=code_only,
                markdown_scores=c.markdown_scores,
                code_scores=c.code_scores)



if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(filetype=args["filetype"],
         archive=args["archive"],
         baseline=args["baseline"],
         code_only=args["codeonly"])
