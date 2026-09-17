import pandas as pd
from .run_spec import RunSpec


def load_grid_search_best_params(data_path, spec: RunSpec,
                                 dataset="Wang2016", save=False):
    dfs = []
    full_path = (
        data_path
        / spec.results_folder
        / spec.results_parent
        / spec.method
        / "GridSearch_CrossSubject"
        / dataset
    )

    for csv_file in full_path.glob("**/best_params_*.csv"):
        try:
            dfs.append(pd.read_csv(csv_file))
        except Exception as e:
            print(f"Read: {csv_file} Error: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["best_score"] = df["best_score"].astype(float)
    df = df.sort_values("best_score", ascending=False).reset_index(drop=True)

    if save:
        out_path = data_path / spec.results_folder / spec.results_parent / spec.method / "all_best_params_summary.csv"
        df.to_csv(out_path, index=False)

    return df