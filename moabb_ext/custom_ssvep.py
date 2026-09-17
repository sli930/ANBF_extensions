""" Pipeline Classes """
import numpy as np
from typing import override
from sklearn.preprocessing import LabelEncoder
from moabb.pipelines.classification import _infer_label_frequencies, _build_sinusoidal_references, _safe_corrcoef, _predict_labels_from_scores, _normalize_score_matrix
from sklearn.utils.validation import check_is_fitted    
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import CCA
from statsmodels.multivariate.cancorr import CanCorr

class SSVEP_FBCCA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_harmonics=3, freq_map=None, n_filters=7, a=1.25, b=0.25, svd=True):
        self.Yf = dict()
        self.svd = svd
        self.cca = None if svd else CCA(n_components=1)
        self.n_harmonics = n_harmonics
        self.freq_map = freq_map
        self.classes_ = []
        self.one_hot_ = {}
        self.class_freqs_ = {}
        self._le, self._slen, self._freqs = None, None, []
        
        self.n_filters = n_filters
        self.a = a
        self.b = b
        if a and b:
            self.weight = [(n+1)**(-a) + b for n in range(n_filters)]
    
    @override
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.slen_ = (X.shape[-1] - 1) / 250
        n_times = X.shape[-1]

        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(
            self.freqs_, dtype=y.dtype if y.dtype != object else object
        )
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class_freqs_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        self.Yf = _build_sinusoidal_references(
            self.class_freqs_, self.n_harmonics, self.slen_, n_times
        )
        return self
    
    @override
    def predict(self, X):
        check_is_fitted(
            self,
            ["freqs_", "classes_", "one_hot_", "slen_", "le_", "class_freqs_"],
        )
        scores = self._score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial, n: self._cca_trial_scores(self.cca, trial, self.Yf, self.classes_, n), 
        )
        return _predict_labels_from_scores(scores, self.classes_)
    
    @override
    def predict_proba(self, X):
        check_is_fitted(
            self,
            ["freqs_", "classes_", "one_hot_", "slen_", "le_", "class_freqs_"],
        )
        scores = self._score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial, n: self._cca_trial_scores(self.cca, trial, self.Yf, self.classes_, n), 
        )
        return _normalize_score_matrix(scores)
    
    def _score_matrix_from_trials(self, X, classes, score_trial_fn):
        data = X # (n_trials, n_ch, n_filters, step)
        band_trials_all = data.transpose(0, 2, 1, 3) # (n_trials, n_filters, n_ch, step)
        
        scores = np.zeros((len(X), self.n_filters, len(classes)))
        for trial_idx, trial in enumerate(X):
            for n in range(self.n_filters):
                band_trial = band_trials_all[trial_idx, n]  # shape (n_ch, step)
                scores[trial_idx, n, :] = score_trial_fn(band_trial, n)
                
        scores = np.sum(scores, axis=1)
        return scores

    def _cca_trial_scores(self, cca, trial, references, classes, n):
        scores = np.zeros(len(classes))
        band_weight = self.weight[n]
        for class_idx, class_label in enumerate(classes):
            if not self.svd:
                S_x, S_y = cca.fit_transform(trial.T, references[class_label].T)
                scores[class_idx] = band_weight * _safe_corrcoef(S_x.ravel(), S_y.ravel()) ** 2
            else:
                can = CanCorr(references[class_label].T, trial.T)
                cancorr = can.cancorr
                scores[class_idx] = band_weight * cancorr[0] ** 2
        return scores

class SSVEP_ANBF(BaseEstimator, ClassifierMixin):
    def __init__(self, freq_map=None, subfreq_freqidx_dict=None, M=0.8, **kwargs):
        super().__init__(**kwargs)
        self.freq_map = freq_map
        self.subfreq_freqidx_dict = subfreq_freqidx_dict
        self.M = M
        self.classes_ = []
        self.one_hot_ = {}
        self.class40_freqs40_ = {}
        self._le, self._slen, self._freqs = None, None, []
        # 每个类别的索引数组和常数
        self._class_indices_ = None        
        
    def fit(self, X, y, sample_weight=None):
        denom = 5 * self.M - 3
        self._const_C = self.M / (denom + 1e-12)
        y = np.asarray(y)
        
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(self.freqs_, dtype=y.dtype if y.dtype != object else object)
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class40_freqs40_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        
        # 把 每个标签label 转换成 对应的频率freq，把 对应的频率 转换成 每个子过滤器对应的频率索引base_idx，把 每个子过滤器对应的频率索引 展开为 临近的五个频率索引indices
        offset = np.array([-2, -1, 0, 1, 2])
        self._class_indices_ = []
        for label in self.classes_:
            freq = self.class40_freqs40_[label]
            base_idx = self.subfreq_freqidx_dict[freq]
            indices = base_idx + offset
            self._class_indices_.append(indices)
        self._class_indices_ = np.array(self._class_indices_)  # shape: (n_classes, 5)        
        return self
    
    def predict(self, X):
        check_is_fitted(self, ["classes_", "_class_indices_", "_const_C"])
        scores = self._compute_score_matrix(X)
        return _predict_labels_from_scores(scores, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "_class_indices_", "_const_C"])
        scores = self._compute_score_matrix(X)
        return _normalize_score_matrix(scores)
    
    def _compute_score_matrix(self, X):
        X = np.squeeze(X, axis=1)                           # (n_trials, n_channels:[O1-O2], n_filters, n_samples) -> (n_trials, n_filters, n_samples)
        trial_energy = np.sum(np.square(X), axis=-1)        # (n_trials, n_filters)
        
        selected = trial_energy[:, self._class_indices_]    # (n_trials, n_classes, 5)
        
        scores = self._compute_score(selected)
        return scores

    def _compute_score(self, matrix):
        main = np.sum(matrix[:, :, 1:4], axis=2)          # (n_trials, n_classes)
        total = np.sum(matrix, axis=2)                    # (n_trials, n_classes)
        
        k = main / np.maximum(total, 1e-12)
        scores = self._const_C * (5 - 3 / k) * main
        return scores

class SSVEP_ANBF_e(SSVEP_ANBF):
    def __init__(self, freq_map=None, subfreq_freqidx_dict=None, M=0.8, main_weight=1, lr_weight=1, **kwargs):
        super().__init__(freq_map=freq_map, subfreq_freqidx_dict=subfreq_freqidx_dict, M=M, **kwargs)
        self.main_weight = main_weight
        self.lr_weight = lr_weight
    
    @override
    def fit(self, X, y, sample_weight=None):
        denom = 5 * self.M - 3
        self._const_C = self.M / (denom + 1e-12)
        y = np.asarray(y)
        
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(self.freqs_, dtype=y.dtype if y.dtype != object else object)
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class40_freqs40_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        
        offset = np.array([-3, -2, -1, 0, 1, 2, 3])
        self._class_indices_ = []
        for label in self.classes_:
            freq = self.class40_freqs40_[label]
            base_idx = self.subfreq_freqidx_dict[freq]
            indices = base_idx + offset
            self._class_indices_.append(indices)
        self._class_indices_ = np.array(self._class_indices_)  # shape: (n_classes, 5)        
        return self
    
    @override
    def _compute_score(self, matrix):
        main_i = (2, 5)
        main = np.sum(matrix[:, :, main_i[0]:main_i[1]], axis=2)            # (n_trials, n_classes)
        mainL = np.sum(matrix[:, :, main_i[0]-1:main_i[1]-1], axis=2)
        mainR = np.sum(matrix[:, :, main_i[0]+1:main_i[1]+1], axis=2)
        
        total_i = (1, 6)
        total = np.sum(matrix[:, :, total_i[0]:total_i[1]], axis=2)         # (n_trials, n_classes)
        totalL = np.sum(matrix[:, :, total_i[0]-1:total_i[1]-1], axis=2)
        totalR = np.sum(matrix[:, :, total_i[0]+1:total_i[1]+1], axis=2)
        
        k = main / np.maximum(total, 1e-12)
        scores = self._const_C * (5 - 3 / k) * main * self.main_weight
        
        kL = mainL / np.maximum(totalL, 1e-12)
        scores += self._const_C * (5 - 3 / kL) * mainL * self.lr_weight
        
        kR = mainR / np.maximum(totalR, 1e-12)
        scores += self._const_C * (5 - 3 / kR) * mainR * self.lr_weight
        return scores

class SSVEP_FBANBF(SSVEP_ANBF):
    def __init__(self, n_banks, freq_map=None, subfreq_freqidx_dict=None, band_weights=[1, 1, 1], M=0.8, **kwargs):
        super().__init__(freq_map=freq_map, subfreq_freqidx_dict=subfreq_freqidx_dict, M=M, **kwargs)
        self.n_banks = n_banks
        self.band_weights = band_weights
        
    @override
    def fit(self, X, y, sample_weight=None):
        denom = 5 * self.M - 3
        self._const_C = self.M / (denom + 1e-12)
        y = np.asarray(y)
        
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(self.freqs_, dtype=y.dtype if y.dtype != object else object)
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class40_freqs40_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        
        # 把 每个标签label 转换成 对应的频率freq，把 对应的频率 转换成 每个子过滤器对应的频率索引base_idx，把 每个子过滤器对应的频率索引 展开为 临近的五个频率索引indices
        offset = np.array([-2, -1, 0, 1, 2])
        self._class_indices_ = []
        for label in self.classes_:
            combine = []
            for n in range(1, self.n_banks + 1):
                freq = self.class40_freqs40_[label]
                base_idx = self.subfreq_freqidx_dict[round(freq * n,1)]
                indices = base_idx + offset
                combine.append(indices)
            self._class_indices_.append(combine)
        self._class_indices_ = np.array(self._class_indices_)  # shape: (n_classes, 5)        
        return self
    
    @override
    def _compute_score(self, matrix):
        scores = np.zeros((len(matrix), len(self.classes_)))
        for h in range(matrix.shape[-2]):                 # Number of harmonic 
            main = np.sum(matrix[:, :, h, 1:4], axis=2)   # (n_trials, n_classes)
            total = np.sum(matrix[:, :, h, :], axis=2)    # (n_trials, n_classes)
            
            k = main / np.maximum(total, 1e-12)
            scores += self.band_weights[h] * (self._const_C * (5 - 3 / k) * main)
        return scores
        
class SSVEP_FBANBF_e(SSVEP_ANBF_e, SSVEP_FBANBF):
    def __init__(self, n_banks, freq_map=None, subfreq_freqidx_dict=None, band_weights=[1, 1, 1], main_weight=1, lr_weight=1, M=0.8, **kwargs):
        super().__init__(n_banks=n_banks,
                         freq_map=freq_map,
                         subfreq_freqidx_dict=subfreq_freqidx_dict,
                         M=M,
                         band_weights=band_weights,
                         main_weight=main_weight,
                         lr_weight=lr_weight,
                         **kwargs
        )
        
    @override
    def fit(self, X, y, sample_weight=None):
        denom = 5 * self.M - 3
        self._const_C = self.M / (denom + 1e-12)
        y = np.asarray(y)
        
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(self.freqs_, dtype=y.dtype if y.dtype != object else object)
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class40_freqs40_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        
        # 把 每个标签label 转换成 对应的频率freq，把 对应的频率 转换成 每个子过滤器对应的频率索引base_idx，把 每个子过滤器对应的频率索引 展开为 临近的五个频率索引indices
        offset = np.array([-3, -2, -1, 0, 1, 2, 3])
        self._class_indices_ = []
        for label in self.classes_:
            combine = []
            for n in range(1, self.n_banks + 1):    
                freq = self.class40_freqs40_[label]
                base_idx = self.subfreq_freqidx_dict[round(freq * n, 1)]
                indices = base_idx + offset
                combine.append(indices)
            self._class_indices_.append(combine)
        self._class_indices_ = np.array(self._class_indices_)  # shape: (n_classes, 5)        
        return self
    
    @override
    def _compute_score_matrix(self, X):
        X = np.squeeze(X, axis=1)                           # (n_trials, n_channels:[O1-O2], n_filters, n_samples) -> (n_trials, n_filters, n_samples)
        trial_energy = np.sum(np.square(X), axis=-1)        # (n_trials, n_filters)
        
        selected = trial_energy[:, self._class_indices_]    # (n_trials, n_classes, 5)
        
        scores = scores = np.zeros((len(selected), len(self.classes_)))
        for h in range(selected.shape[-2]):
            h_matrix = selected[:, :, h, :]
            scores += self.band_weights[h] * SSVEP_ANBF_e._compute_score(self, h_matrix)
        return scores

# Experiments
'''
class SSVEP_ANBF_sumLR_sameK(SSVEP_ANBF):
    def __init__(self, freq_map=None, subfreq_freqidx_dict=None, M=0.8, main_weight=1, lr_weight=1):
        super().__init__(freq_map, subfreq_freqidx_dict, M)
        self.main_weight = main_weight
        self.lr_weight = lr_weight
    
    @override
    def _compute_score(self, matrix):
        main = np.sum(matrix[:, :, 1:4], axis=2)          # (n_trials, n_classes)
        mainL = np.sum(matrix[:, :, 1-1:4-1], axis=2)
        mainR = np.sum(matrix[:, :, 1+1:4+1], axis=2)
        total = np.sum(matrix, axis=2)                    # (n_trials, n_classes)
        
        k = main / np.maximum(total, 1e-12)
        scores = self._const_C * (5 - 3 / k) * main * self.main_weight
        scores += self._const_C * (5 - 3 / k) * mainL * self.lr_weight
        scores += self._const_C * (5 - 3 / k) * mainR * self.lr_weight
        return scores
    
class SSVEP_ANBF_sumLR_diffK(SSVEP_ANBF):
    def __init__(self, freq_map=None, subfreq_freqidx_dict=None, M=0.8, main_weight=1, lr_weight=1):
        super().__init__(freq_map, subfreq_freqidx_dict, M)
        self.main_weight = main_weight
        self.lr_weight = lr_weight
    
    @override
    def _compute_score(self, matrix):
        main = np.sum(matrix[:, :, 1:4], axis=2)          # (n_trials, n_classes)
        mainL = np.sum(matrix[:, :, 1-1:4-1], axis=2)
        mainR = np.sum(matrix[:, :, 1+1:4+1], axis=2)
        total = np.sum(matrix, axis=2)                    # (n_trials, n_classes)
        
        k = main / np.maximum(total, 1e-12)
        scores = self._const_C * (5 - 3 / k) * main * self.main_weight
        
        kL = mainL / np.maximum(total, 1e-12)
        scores += self._const_C * (5 - 3 / kL) * mainL * self.lr_weight
        
        kR = mainR / np.maximum(total, 1e-12)
        scores += self._const_C * (5 - 3 / kR) * mainR * self.lr_weight
        return scores
'''