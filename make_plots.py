
import config
import pandas as pd
from plotting.loaders import load_all, load_one, load_folder, RunSpec, load_confusion_matrix, load_grid_search_best_params
from plotting.table_plots import get_stats, get_ttest
from plotting.metric_plots import plot_metric_static, plot_metric_vs_window_length, plot_metric_vs_n_classes, plot_heatmap
from plotting.matrix_plots import plot_matrix
from plotting.stats import summarize_counts

PARADIGMS = {
    "CCA":   "SSVEP",
    "FBCCA": "CustomFilterBankSSVEP",
    "ANBF":  "CustomFilterBankSSVEP_ANBF",
}
data_path = config.current_cache_data_path

# python.exe -m scripts.make_plots
def main():
    pass

def anbf_e_best_params():
    run = RunSpec("test_tmax_n40", "tmax3.64_n40", "ANBF_e_GRID")
    df = load_grid_search_best_params(data_path, run)
    summarize_counts(df, ["ssvep_anbf_e__main_weight", "ssvep_anbf_e__lr_weight"])

def fbanbf_best_params():
    run = RunSpec("test_tmax_n8", "tmax5.5_n8", "FBANBF_GRID")
    df = load_grid_search_best_params(data_path, run)
    summarize_counts(df, ["ssvep_fbanbf__band_weights"])

def fbanbf_cm():
    # FBANBF - ANBF, n = 8
    _, df1_percent = load_confusion_matrix(data_path, "three_methods_tmax_n8_predictions/ANBF/tmax5.5_n8.txt")
    plot_matrix(df1_percent, "ANBF (8 Classes, 5s)")
    _, df2_percent = load_confusion_matrix(data_path, "all_anbf_fixed_grid_params_tmax_n8/tmax5.5_n8_predictions/FBANBF/FBANBF.txt")
    plot_matrix(df2_percent, "FBANBF_Fixed (8 Classes, 5s) Fixed")

    df_diff = df2_percent - df1_percent
    plot_matrix(df_diff, "FBANBF_Fixed - ANBF (8 Classes, 5s)", diff=True)

def anbf_e_cm():
    # ANBF_e - ANBF, n = 40
    _, df1_percent = load_confusion_matrix(data_path, "three_methods_tmax_n40_predictions/tmax3.64_n40.txt")
    plot_matrix(df1_percent, "ANBF (40 Classes, 3s)", svg=False)
    _, df2_percent = load_confusion_matrix(data_path, "all_anbf_fixed_grid_params_tmax_n40/tmax3.64_n40_predictions/ANBF_e/ANBF_e.txt")
    plot_matrix(df2_percent, "ANBF_e_Fixed (40 Classes, 3s) ", svg=False)

    df_diff = df2_percent - df1_percent
    plot_matrix(df_diff, "ANBF_e_Fixed - ANBF (40 Classes, 3s)", diff=True, svg=False)

def fbanbf_heatmap():
    df = load_folder(data_path, "fbanbf_grid_secondW_thirdW_n8_tmax5.5", "FBANBF", "CustomFilterBankSSVEP_ANBF", 5.5)
    stats = df.groupby(["w_2", "w_3"])["Accuracy"].mean().round(4)
    plot_heatmap(stats, "Third Filter Bank Weight", "Second Filter Bank Weight")

def anbf_e_heatmap():
    df = load_folder(data_path, "anbf_e_grid_m_lr_n40_tmax3.64", "ANBF_e", "CustomFilterBankSSVEP_ANBF", 3.64)
    stats = df.groupby(["w_c", "w_lr"])["Accuracy"].mean().round(4)
    plot_heatmap(stats, "Side Weight", "Main Weight")

def tmax_3m_n_arr_itr():
    dfs = []
    for method, paradigm in PARADIGMS.items():
        df = load_folder(data_path, "three_methods_n_tmax5.5", method, paradigm)
        dfs.append(df)
    
    df_arr, df_itr = get_stats(pd.concat(dfs, ignore_index=True))
    plot_metric_vs_n_classes(df_arr)
    plot_metric_vs_n_classes(df_itr, "ITR (bits/min)", y_step=10)

def n8n40_3m_tmax_arr_itr():
    dfs = []
    for method, paradigm in PARADIGMS.items():
        df = load_folder(data_path, "three_methods_tmax_n8", method, paradigm)
        dfs.append(df)
        
        df = load_folder(data_path, "three_methods_tmax_n40", method, paradigm)
        dfs.append(df)
        
    df_arr, df_itr = get_stats(pd.concat(dfs, ignore_index=True))
    plot_metric_vs_window_length(df_arr)
    plot_metric_vs_window_length(df_itr, "ITR (bits/min)", y_step=10)

def n8n40_3m_arr_itr():
    def load_group(n):
        dfs = []
        for method, spec in RUN_GROUPS[n].items():
            df = load_one(data_path, spec, PARADIGMS[method])
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    RUN_GROUPS = {
        8: {
            "CCA":   RunSpec("three_methods_tmax_n8",  "tmax5.5_n8",  "CCA"),
            "FBCCA": RunSpec("three_methods_tmax_n8",  "tmax5.5_n8",  "FBCCA"),
            "ANBF":  RunSpec("three_methods_tmax_n8",  "tmax5.5_n8",  "ANBF"),
        },
        40: {
            "CCA":   RunSpec("three_methods_tmax_n40", "tmax5.5_n40", "CCA"),
            "FBCCA": RunSpec("three_methods_tmax_n40", "tmax5.5_n40", "FBCCA"),
            "ANBF":  RunSpec("three_methods_tmax_n40", "tmax5.5_n40", "ANBF"),
        },
    }

    df_n8  = load_group(8)
    df_n40 = load_group(40)

    get_stats(pd.concat([df_n8, df_n40], ignore_index=True))
    plot_metric_static(pd.concat([df_n8, df_n40], ignore_index=True))

def n8n40_ttest():
    # n = 8
    base_run = RunSpec("three_methods_tmax_n8", "tmax5.5_n8", "ANBF")    
    target_runs = [
        RunSpec("test_tmax_n8", "tmax5.5_n8/ANBF_e_GRID", "ANBF_e_GRID"),
        RunSpec("test_tmax_n8", "tmax5.5_n8/FBANBF_GRID", "FBANBF_GRID") ########################################
        ]
    
    # # n = 40
    # base_run = RunSpec("three_methods_tmax_n40", "tmax3.64_n40", "ANBF")
    # target_runs = [
    #     RunSpec("test_tmax_n40", "tmax3.64_n40/ANBF_e_GRID", "ANBF_e_GRID")
    #     ]
    
    base_df = load_one(data_path, base_run, "CustomFilterBankSSVEP_ANBF")
    for run in target_runs:
        target_df = load_one(data_path, run, "CustomFilterBankSSVEP_ANBF")
        get_stats(pd.concat([base_df, target_df], axis=0, ignore_index=True))
        get_ttest(base_df, target_df)

def n8n40_anbf_acc_itr():
    # # n = 8, tmax 5.5
    # runs = [
    #     RunSpec("three_methods_tmax_n8", "tmax5.5_n8", "ANBF"),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax5.5_n8", "ANBF_e_Fixed", index=0),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax5.5_n8/FBANBF", "FBANBF_Fixed"),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax5.5_n8", "FBANBF_e_Fixed", index=1),
    # ]
    
    # # n = 8, tmax = 3.64
    # runs = [
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax3.64_n8/ANBF", "ANBF"), # RunSpec("three_methods_tmax_n8", "tmax3.64_n8", "ANBF")
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax3.64_n8/ANBF_e", "ANBF_e_Fixed"),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax3.64_n8/FBANBF", "FBANBF_Fixed"),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n8", "tmax3.64_n8/FBANBF_e", "FBANBF_e_Fixed")
    # ]

    # # n = 40, tmax = 3.64 
    # runs = [
    #     RunSpec("three_methods_tmax_n40", "tmax3.64_n40", "ANBF"),
    #     RunSpec("anbf_e_grid_m_lr_n40_tmax3.64", "m1.00_lr0.40", "ANBF_e_Fixed", 3.64),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n40", "tmax3.64_n40/FBANBF", "FBANBF_Fixed"),
    #     RunSpec("all_anbf_fixed_grid_params_tmax_n40", "tmax3.64_n40/FBANBF_e", "FBANBF_e_Fixed"),
    # ]

    # n = 40, tmax = 1.64
    runs = [
        RunSpec("all_anbf_fixed_grid_params_tmax_n40", "tmax1.6400000000000001_n40/ANBF", "ANBF"),
        RunSpec("all_anbf_fixed_grid_params_tmax_n40", "tmax1.6400000000000001_n40/ANBF_e", "ANBF_e_Fixed"),
        RunSpec("all_anbf_fixed_grid_params_tmax_n40", "tmax1.6400000000000001_n40/FBANBF", "FBANBF_Fixed"),
        RunSpec("all_anbf_fixed_grid_params_tmax_n40", "tmax1.6400000000000001_n40/FBANBF_e", "FBANBF_e_Fixed"),
    ]

    dfs = load_all(data_path, runs, "CustomFilterBankSSVEP_ANBF")
    get_stats(dfs)

if __name__ == "__main__":
    main()