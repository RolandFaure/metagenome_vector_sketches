import pandas as pd

def latex_escape(text):
    """Escape LaTeX special characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def csv_to_latex_table(
    csv_file,
    output_file,
    caption="Caption",
    label="tab:my_table",
    font_size="\\footnotesize"
):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Alignment: left-align every column
    col_format = "@{}" + "l" * len(df.columns) + "@{}"

    with open(output_file, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write(font_size + "\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + col_format + "}\n")
        f.write("\\toprule\n")

        # Header
        headers = " & ".join(latex_escape(col) for col in df.columns)
        f.write(headers + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for _, row in df.iterrows():
            values = []
            for val in row:
                if isinstance(val, float):
                    values.append(f"{val:.3f}".rstrip("0").rstrip("."))
                else:
                    values.append(latex_escape(val))
            f.write(" & ".join(values) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {output_file}")


import sys
fn = sys.argv[1]
csv_to_latex_table(
    fn,
    "table.tex",
    caption="Statistics of the generated matrices.",
    label="tab:matrix_stats",
)