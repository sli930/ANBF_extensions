from sklearn.metrics import confusion_matrix

def load_confusion_matrix(data_path, file_path):
    y_true, y_pred = [], []
    with open(data_path / file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, p = line.split('\t')
            y_true.append(int(t))
            y_pred.append(int(p))

    cm_counts = confusion_matrix(y_true, y_pred)
    # Calculate Percentage
    cm_percent = cm_counts.astype('float') / cm_counts.sum(axis=1, keepdims=True) * 100
    return cm_counts, cm_percent