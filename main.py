import argparse
from common import *

parser = argparse.ArgumentParser(prog="PlagiarismChecker",
                                 description="Runs the workflow for checking Plagiarism")

parser.add_argument("-f", "--filetype", type=str)
parser.add_argument("-a", "--archive", type=str)
parser.add_argument("-b", "--baseline", type=str, default=None)
parser.add_argument("-c", "--codeonly", type=bool, default=True)
parser.add_argument("-e", "--extract_name", type=str, default="None")

parser.add_argument("-t", "--threshold", type=float, default=0.005)
parser.add_argument("-co", "--cutoff", type=float, default=0.8)
parser.add_argument("-bt", "--barrenthreshold", type=float, default=None)

parser.add_argument("-an", "--assignmentname", type=str, default=None)
parser.add_argument("-cn", "--coursename", type=str, default=None)


def main(arguments: vars) -> None:
    """
    Run the main script to flag plagiarism.

    :param arguments: The arguments for the plagiarism checker.
    """
    print("Data Extraction")
    dd = DataDict(arguments=arguments)
    dd.get_data()

    print("File Comparison")
    c = CompareDict(arguments=arguments,
                    data_dict=dd.data_dict,
                    all_frequency_values=dd.all_frequency_values)
    c.run_comparison()

    print("Plagiarism Flagging")
    code_only = True if "ipynb" not in arguments["filetype"] else arguments[
        "codeonly"]  # dont allow markdown comparison for non-notebook files
    f = Flagger(arguments=arguments,
                file_names=list(dd.data_dict.keys()),
                code_only=code_only,
                markdown_scores=c.markdown_scores,
                code_scores=c.code_scores,
                code_slices=c.code_slices)

    f.flag_submissions()
    f.save_csv()

    print("Reporting")
    r = Report(arguments=arguments,
               flagged_df=f.flagging_df,
               file_names=list(dd.data_dict.keys()))
    r.generate_report()
    print("DONE!")


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
