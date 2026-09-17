

def summarize_counts(df, columns: list[str] | None = None, score_col="best_score"):
    print(df.head())
    if columns:
        counts = df.groupby(columns).size().sort_values(ascending=False)
        print("Hyper-Parameter Set Count:\n", counts)
        print("Average Accuracy:", df[score_col].mean())
        return counts
