## MimeoCode Plagiarism Checking Pipeline
Here you can automatically generate the report for submissions.

#### Some important notes:
- The submissions should be contained in a ZIP folder.
- Submission names should be unique!
- If a baseline is used, it should also be contained in the submission zip.
- Running the script can take a few minutes depending on amount of submissions and your local machine.

#### How to use:
To run the main script make sure MikTex or another latex kernel is installed on your machine.
Also make sure you have all requirements found in `requirements.txt` in your venv.

To run the main script use: `python main.py`
However you also want to use the arguments to provide files, and other parameters.

#### The arguments available:

*Provide Files:*
- `-f` or `--filetype`: Set the file type thta should be checked for plagiarism. Example for jupyter notebooks: `-f ".ipynb"`.
- `-a` or `--archive`: Point to the archive the submissions are contained in. Example: `-a "submissions.zip"`.
- `-b` or `--baseline`: If a baseline is available, you can determine it here by giving the file name (The file extension is not needed.) Example: `-b "baseline_file"`.

*Manipulate flaggig process:*
- `-c` or `--codeonly`: If you only want to compare code rater than markdown aswell. (Only applicable for notebooks.) Example: `-c True`.
- `-e` or `--extract_name`: If names should be extracted from the filenames. There are default rules available for `canvas`and `code_grade`. Example: `-e "canvas"`
- `-t` or `--threshold`: Set the threshold for flagging. **Default is 0.005**. Example: `-t 0.01`.
- `-co` or `--cutoff`: To set a specific cutoff for similarity for better case extraction. **Default is 0.8**. Example: `-co 0.75`.
- `-bt` or `--barrenthreshold`: Set a specific threshold for assignments that are too close to the baseline for exclusion. Example: `-bt 0.8`. 

*Tune the final Report:*
- `-an` or `--assignmentname`: To set the Title of the report. Example: `-an "Example Assignment"`.
- `-cn` or `--coursename`: To set the course name on the report (subtitle). Exaxmple: `-cn "Example Course".

*A final example command could look like:*
`python main.py -f ".ipynb" -a "submssions.zip" -b "assignment_baseline" -e "canvas" -an "Assignment X" -cn "The Course" -bt 0.8 -c False` 