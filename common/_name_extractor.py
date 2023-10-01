def canvas_rule(x: str) -> str:
    lst = x.split("_")
    try:
        if lst[1] == "LATE":
            return "_".join([lst[1],lst[-1][:7]])
        return lst[-1][:7]
    except:
        return x

def code_grade_rule(x: str) -> str: #5016643 - F.A. Moser (Ferris)_Exercise-1.
    cand = x.split("_")[0]
    cand = cand.split(" - ")[-1]
    return cand
