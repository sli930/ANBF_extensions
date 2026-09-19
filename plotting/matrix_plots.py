import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def plot_matrix(df_cm, title, diff=False, svg=False, figsize=(10, 8), csv=False):
    cmap = LinearSegmentedColormap.from_list('GreenWhiteRed', ['green', 'white', 'red']) if diff else "Blues" 
    norm = TwoSlopeNorm(vmin=df_cm.min(), vcenter=0, vmax=df_cm.max()) if diff else None
    
    # 获取类别数
    n_classes = df_cm.shape[0]
    if n_classes == 8:            
        freq_map = {0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0, 4: 14.0, 5: 15.0, 6: 8.0, 7: 9.0}
        plt.figure(figsize=figsize)
        
        df_cm = pd.DataFrame(df_cm, index=range(1,9), columns=range(1,9))
        new_index = list(range(7,9)) + list(range(1,7))
        df_cm = df_cm.reindex(index=new_index, columns=new_index)
        
    else:            
        freq_map = {0: 10.0, 1: 10.2, 2: 10.4, 3: 10.6, 4: 10.8, 5: 11.0, 6: 11.2, 7: 11.4, 8: 11.6, 9: 11.8, 10: 12.0, 11: 12.2, 12: 12.4, 13: 12.6, 14: 12.8, 15: 13.0, 16: 13.2, 17: 13.4, 18: 13.6, 19: 13.8, 20: 14.0, 21: 14.2, 22: 14.4, 23: 14.6, 24: 14.8, 25: 15.0, 26: 15.2, 27: 15.4, 28: 15.6, 29: 15.8, 30: 8.0, 31: 8.2, 32: 8.4, 33: 8.6, 34: 8.8, 35: 9.0, 36: 9.2, 37: 9.4, 38: 9.6, 39: 9.8}
        plt.figure(figsize=(30, 25))
        
        df_cm = pd.DataFrame(df_cm, index=range(1,41), columns=range(1,41))
        new_index = list(range(31,41)) + list(range(1,31))
        df_cm = df_cm.reindex(index=new_index, columns=new_index)
    
    
    df_cm_labeled = df_cm.copy()
    df_cm_labeled.index = [freq_map[i-1] for i in new_index]
    df_cm_labeled.columns = [freq_map[i-1] for i in new_index]
    print(df_cm_labeled)
    if csv:
        df_cm_labeled.round(1).to_csv(f'{title}_cm.csv', float_format='%.1f')
        
    ax = sns.heatmap(df_cm, annot=True, fmt='.1f', cmap=cmap, norm=norm,
        xticklabels=[freq_map[i-1] for i in new_index], 
        yticklabels=[freq_map[i-1] for i in new_index],
        annot_kws={'size': 12}
        )
    for i in range(n_classes):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))
    plt.xlabel('Predicted Class (Hz)')
    plt.ylabel('True Class (Hz)')
    plt.title(f'{title} Confusion Matrix (Row Normalized %)')
    if svg:
        plt.savefig(f'{title}.svg', format='svg', bbox_inches='tight')
    plt.show()