def canvas_rule(x: str) -> str:
    """
    Format the file names to fit the canvas formatting.

    Format: lastname_studentnumber_nbr_filename

    :param x: The string.
    """
    lst = x.split("_")
    try:
        if lst[1] == "LATE":
            return "_".join([lst[1], lst[2]])
        return lst[1]
    except:
        return x


def code_grade_rule(x: str) -> str:  # 5016643 - F.A. Moser (Ferris)_Exercise-1.
    cand = x.split("_")[0]
    cand = cand.split(" - ")[-1]
    return cand
