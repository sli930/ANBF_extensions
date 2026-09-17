import numpy as np
from scipy import stats

def get_ttest(base_df, target_df):
    diff = target_df['Accuracy']*100 - base_df['Accuracy']*100
    improvements = diff.dropna().tolist()

    t_stat, p_value_two_tailed = stats.ttest_1samp(improvements, 0)
    p_value_one_tailed = p_value_two_tailed / 2   # only consider whether the improvement is greater than 0
    
    print(f"{target_df["Method"].iloc[0]} VS {base_df["Method"].iloc[0]}, t = {t_stat:.3f}, p (two-tailed) = {p_value_two_tailed:.4f}")
    if p_value_one_tailed < 0.05:
        print("improvment > 0, the result has statistical significance")
    else:
        print("improvment ≈ 0, the result does not have a significant different")

def get_stats(dfs):
    return get_accuracy(dfs), get_itr(dfs)

def get_accuracy(dfs):    
    output = getMeanStd(dfs, x="Method", y="Accuracy")
    print(output.map(lambda v: f"{v:.4g}"))
    return dfs

def get_itr(dfs):
    def compute_itr(row):
        N = row['n_classes']
        P = row['Accuracy']
        T = row['Window length (s)'] + 0.5  # 计算ITR，添加 0.5s的 注视移动时间
        
        # 处理边界情况
        if P == 0:
            B = 0.0
        elif P == 1:
            B = np.log2(N)
        else:
            B = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
        
        return B * (60 / T)
    
    dfs['ITR (bits/min)'] = dfs.apply(compute_itr, axis=1)
    output = getMeanStd(dfs, x="Method", y="ITR (bits/min)")
    print(output.map(lambda v: f"{v:.4g}"))
    return dfs


def getMeanStd(dfs, x, y):        
    return dfs.groupby([x, "n_classes", "Window length (s)"])[y].agg(['mean', 'std'])