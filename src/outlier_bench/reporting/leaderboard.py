from __future__ import annotations
import pandas as pd
import numpy as np


def csv_to_markdown(csv_path: str, md_path: str, sort_by: str = "pr_auc", top_n: int = 20):
    df = pd.read_csv(csv_path)

    # Replace NaN with empty string for readability
    df = df.replace({np.nan: ""})

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df_out = df.head(top_n)
    md = df_out.to_markdown(index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
        f.write("\n")
