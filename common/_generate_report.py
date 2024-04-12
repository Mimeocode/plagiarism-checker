import shutil
import subprocess
import os
import sys
import pandas as pd
from attrs import define, field,setters
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from tqdm import tqdm

EPS = 0.001
@define(kw_only=True)
class Report:
    arguments: vars = field(on_setattr=setters.frozen)
    flagged_df: pd.DataFrame = field(on_setattr=setters.frozen)
    file_names: list = field(on_setattr=setters.frozen)

    working_dir: str = ""
    code_only: bool = field(init=False)

    def generate_report(self):
        assignment_name = self.arguments["assignmentname"]
        self.code_only = self.arguments["codeonly"]

        self.working_dir = f"./assets/report_{assignment_name}_{hash(assignment_name)}"
        shutil.copytree("./assets/report_template", self.working_dir)

        self._plot_weighted_network()
        insertion_dir = {
            "title": assignment_name,
            "course": self.arguments["coursename"],
            "numofsub": str(len(self.file_names)),
            "numdep": str(len(self.flagged_df)),
            "numofplag": str(len(self.flagged_df[self.flagged_df['Classification'] > 0])),
            "plagthresh": str(self.arguments["threshold"]),
            "barrthresh": str(self.arguments["barrenthreshold"]),
            "distrsect": self._get_distr_section(),
            "cldiscifcodeonly": r"For the analysis of this submission \textit{markdown} and \textit{combined} can be disregarded, since no textual answers were asked in the notebooks.\\"
                                              if self.code_only else "",
            "casetable": self._get_table(),
            "cutoff": self.arguments["cutoff"],
            "cases": self._generate_cases(),
        }

        with open(f"{self.working_dir}/main.tex", "r+", encoding="utf-8") as f:
            content = f.read()

            f.seek(0)  # allows for overwrite
            f.write(content%insertion_dir)  # this inserts dict items into %(key)s
            f.truncate()

        os.chdir(self.working_dir)  # go into working dir to execute rendering --> necessary to find images
        proc = subprocess.Popen(["pdflatex", f"main.tex"])
        proc.communicate()
        proc = subprocess.Popen(["pdflatex", f"main.tex"])
        proc.communicate()

        #self._rm_working_dir()

    def _get_distr_section(self) -> str:
        self._plot_specifc_dist()  # generating plots on the go to allow for specific plottings

        if "ipynb" in self.arguments["filetype"]:
            self._plot_distr()  # is only used for notebooks

            distr_section = r"Below, in Fig. \ref{fig:f1}, we present a graphical representation depicting the distribution of similarities for both code and markdown. --#"
            if self.code_only:
                distr_section = distr_section.replace("--#", r"Given that, for the current submissions, markdown can be considered irrelevant in the context of measuring plagiarism, a dedicated figure is provided below for the assessment of code similarity (Fig. \ref{fig:f2}). --#")
            else:
                distr_section = distr_section.replace(r"--#", r"To enhance clarity and visual distinction, an independent figure focusing solely on the code is presented below (Fig. \ref{fig:f2}). --#")
            distr_section = distr_section.replace(r"--#", r"""\begin{figure}[!hbp]
  \centering
  \subfloat[Distribution of Scores for Markdown and Code]{\includegraphics[width=0.4\textwidth]{img/distr_plot.png}\label{fig:f1}}
  \hfill
  \subfloat[Distribution of Scores for Code]{\includegraphics[width=0.4\textwidth]{img/specific_distr_plot.png}\label{fig:f2}}
  \caption{Distribution Overview}
\end{figure}\\)""")
        else:
            distr_section = r"""Fig. \ref{fig:f1} below shows the distribution of code. \begin{figure}[!hbp]
  \centering
  \includegraphics[width=0.6\textwidth]{img/specific_distr_plot.png}
  \label{fig:f1}
  \caption{Distribution Overview}
\end{figure}\\"""
        return distr_section

    def _get_table(self) -> str:
        def __replace_items(string: str, repl_dict: dict) -> str:
            for key, item in repl_dict.items():
                string = string.replace(key, item)
            return string
        sort_col = "Metric_1" if self.code_only else "Metric_2"
        ltx = self.flagged_df[self.flagged_df["Classification"] > 0].copy()
        del ltx["Classification"]
        del ltx["Code_Slices"]
        ltx = ltx.sort_values(sort_col, ascending=False).to_latex(index=False, float_format="{:.3f}".format)

        repl_dict = {
            r"\begin{tabular}{llrrr}": r"\begin{longtable}{p{.28\textwidth} p{.28\textwidth} p{.10\textwidth} p{.08\textwidth} p{.10\textwidth}}",
            r"\end{tabular}": r"\end{longtable}",
            r"\toprule": "",
            r"\midrule": "",
            r"\bottomrule": "",
            "Metric_0": "markdown",
            "Metric_1": "code",
            "Metric_2": "combined",
            "_": r"\_",}

        ltx = __replace_items(ltx, repl_dict)

        ltx = ltx.split("\n")
        ltx.insert(2, r"\hline")
        ltx.insert(4, r"\hline\hline")
        ltx.insert(-3, r"\hline")
        ltx = "\n".join(ltx)
        return ltx

    def _generate_cases(self) -> str:
        def __format_code(paragraph:str) -> str:
            #highlight = paragraph.split(">>")
            #highlighted = ""
            #for i in range(len(highlight)//2):
            #    first = highlight[i]
            #    second, third = highlight[i+1].split("<<")
            #    highlighted += r"|\colorbox{green}{"+"".join([first, second])+r"}|"+third
            return "\n".join([x for x in paragraph.split("\n") if not x.startswith("#") and x.strip()])

        case_string = ""
        selection = self.flagged_df[(self.flagged_df["Classification"] > 0) &
                                    (self.flagged_df["Metric_1"] > self.arguments["cutoff"]) &
                                    (self.flagged_df["Metric_0"] > self.arguments["cutoff"])]
        selection = selection.sort_values("Metric_1", ascending=False)
        for _, case in tqdm(selection.iterrows()):
            case_string += r"\subsection*{\textbf{Case}: "+case["Submission 1"]+" and "+case["Submission 2"]+r"}\\"
            code1, code2 = case["Code_Slices"]
            code1, code2 = __format_code(code1), __format_code(code2)
            case_string += r"\textbf{"+case["Submission 1"]+r"}\\ \begin{python}"+"\n"+code1+"\n"+r"\end{python}\\"
            case_string += r"\textbf{"+case["Submission 2"]+r"}\\ \begin{python}"+"\n"+code2+"\n"+r"\end{python}\newpage"

        return case_string

    # plotting

    def _plot_distr(self) -> None:
        """
        Plots distribution of submissions similarity
        :param df: dataframe of submission daza
        :param print_top: print the "n" most similar pairs
        :return:
        """
        plot_df = self.flagged_df.copy()
        plot_df["Classification"] = "Unsuspected"  # TODO : should be value based not str
        plot_df[self.flagged_df["Classification"] == 1]["Classification"] = "Suspected"
        plot_df[self.flagged_df["Classification"] > 1]["Classification"] = "Likely"

        ax = sns.jointplot(data=plot_df,
                           x="Metric_0",
                           y="Metric_1",
                           kind="scatter",
                           space=0,
                           hue="Classification",
                           hue_order=['Unsuspected', 'Suspected', 'Likely'])

        sns.move_legend(ax.ax_joint, "upper left", bbox_to_anchor=(1, 1.2))
        plt.tight_layout()
        plt.savefig(f"{self.working_dir}/img/distr_plot.png")

    def _plot_specifc_dist(self, metric="Metric_1") -> None:
        sns.displot(data=self.flagged_df, x=metric, fill=True)
        thresh = self.flagged_df[metric].quantile(1-self.arguments["threshold"])
        co = self.arguments["cutoff"]
        plt.axvline(thresh, color="orange", label=f"top {self.arguments['threshold']:.2%}")
        plt.axvline(co, color="red", label=f"{co:.2%}-Cutoff")
        plt.legend()
        plt.savefig(f"{self.working_dir}/img/specific_distr_plot.png")

    def _get_top_percentile(self, df: pd.DataFrame, weight_col, percentile) -> pd.DataFrame:
        df = df.copy()
        # Find out percentiles
        lower = np.percentile(df[weight_col], 100 - percentile)
        upper = np.percentile(df[weight_col], 100)
        # Select data between
        top_percentile_df = df[df[weight_col].between(lower, upper)]
        return top_percentile_df

    def _plot_weighted_network(self,
                               percentile: int = 100,
                               use_graphviz: bool = False,
                               anonymous: bool = False,) -> None:

        metric_col = "Metric_1" if self.code_only or self.arguments["filetype"] != "ipynb" else "Metric_2"
        suspects = self.flagged_df[self.flagged_df["Classification"] > 0]

        df_100_c = self._get_top_percentile(suspects.copy(), metric_col, 100)[[metric_col]]
        df = self._get_top_percentile(suspects, metric_col, percentile)  # .sort_values(weight_col, ascending=True)
        df = df[df["Metric_1"] > self.arguments["cutoff"]]
        def _sigmoid(target_df: pd.DataFrame, x: float) -> float:
            # super complicated sigmoid modification that does what I wanted lol (this took me 4 days to make:)
            t = np.average([target_df.min(), target_df.max()])
            b = np.power(t, 10 * t) * 100
            f = 1 / (1 + np.exp(b * (-x + t)))
            return f

        G = nx.Graph()  # Create an empty graph
        fig_dim = np.sqrt(len(df)) + 5
        plt.figure(1, figsize=(fig_dim, fig_dim), dpi=200)
        for index, row in df.iterrows():  # Add edges and their weights to the graph
            G.add_edge(row["Submission 1"].encode(sys.stdout.encoding, "replace"), # TODO: the encoding stuff is shit
                       row["Submission 2"].encode(sys.stdout.encoding, "replace"),
                       weight=str(row[metric_col]),
                       len="10")

        edge_weights = [float(G[u][v]['weight']) for u, v in
                        G.edges()]  # Create a list of edge weights for visualization
        scale_factor = 4.  # Set the edge thickness scaling factor
        pos = nx.nx_pydot.graphviz_layout(G, prog="neato") if use_graphviz else nx.spring_layout(G, seed=42,
                                                                                                 k=1.5 * 1 / np.sqrt(
                                                                                                     len(G.nodes())),
                                                                                                 iterations=20)

        nx.draw_networkx(G, pos,
                         with_labels=(not anonymous),
                         node_color='pink',
                         node_size=600,
                         font_size=6,
                         font_weight='bold',
                         edge_color="#FFFFFF")

        nx.draw_networkx_edges(G, pos,
                               edgelist=G.edges(),
                               width=[_sigmoid(df_100_c, weight) * scale_factor for weight in edge_weights],
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.rainbow,
                               alpha=[(weight - EPS) ** 2 for weight in edge_weights],
                               edge_vmin=df_100_c.min())

        labels = nx.get_edge_attributes(G, 'weight')
        for k, v in labels.items():
            labels[k] = f"{float(v):.3f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6, alpha=0.8)
        plt.savefig(f"{self.working_dir}/img/network.png", format="PNG")




    def _rm_working_dir(self) -> None:
        shutil.rmtree(self.working_dir)
