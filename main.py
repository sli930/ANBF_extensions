import warnings
import inspect
import moabb
import config
import numpy as np
from joblib import Parallel, delayed
from moabb.datasets import Wang2016
from moabb.paradigms import SSVEP
from moabb.pipelines import SSVEP_CCA 
from moabb.evaluations import CrossSubjectEvaluation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from moabb_ext.custom_ssvep import SSVEP_FBCCA, SSVEP_ANBF, SSVEP_ANBF_e, SSVEP_FBANBF, SSVEP_FBANBF_e
from moabb_ext.custom_paradigm import CustomFilterBankSSVEP, CustomFilterBankSSVEP_ANBF

moabb.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cross_decomposition")

default_cache_config = {    
    "save_raw": True,
    "save_epochs": False,
    "save_array": False,
    "use": True,    
    "overwrite_raw": False,
    "overwrite_epochs": False,
    "overwrite_array": False,
    "path": None,
}

current_cache_data_path = config.current_cache_data_path
current_cache_data_path.mkdir(parents=True, exist_ok=True)

class SSVEP_Test:
    def __init__(self):
        dataset = Wang2016()
        
        # dataset.subject_list = dataset.subject_list[:2]
        
        self.datasets = [dataset]
        self.selected_channels = ['O1', 'O2']
        self.print_review = True
        self.tmin = 0.5 + 0.14 # flicking stimulus start at 0.5s

    def main(self):
        ''' Playground 
        # n = 8
        # tmax = 1.0
        # freq_map = self._get_freqs_map(n)
        
        # self.CCA(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=12)
        # self.FBCCA(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=4, n_filters=7, svd=True)
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1)
        
        # n = None
        # self.CCA(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=12)
        # self.FBCCA(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=4, n_filters=7, svd=True)
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1)
        
        # tmax = None
        # self.CCA(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=12)
        # self.FBCCA(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=4, n_filters=7, svd=True)
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1)
    
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1)
        # self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1, n_filters=2)
        # self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1, n_filters=3)
        '''
        ''' tmax: 0.5 -> 5.5, n = 8, 40 '''        
        # for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        #     for n in [40, 8]:
        #         tmax = min(5.5, self.tmin + t)
        #         results_parent_path = current_cache_data_path / f"three_methods_tmax_n{n}" / f"tmax{tmax}_n{n}"
        #         freq_map = self._get_freqs_map(n)

        #         self.CCA(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=12, results_parent_path=results_parent_path)
        #         self.FBCCA(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=5, results_parent_path=results_parent_path, n_filters=7)     
        #        self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path)
        
        ''' n: 8 -> 40, tmax = 5.5 '''
        # for n in [8, 16, 24, 32, 40]:
        #     tmax = 5.5
        #     results_parent_path = current_cache_data_path / f"three_methods_n_tmax{tmax}" / f"tmax{tmax}_n{n}"
        #     freq_map = self._get_freqs_map(n)
        #     self.CCA(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=12, results_parent_path=results_parent_path)
        #     self.FBCCA(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=7, results_parent_path=results_parent_path, n_filters=7) 
        #     self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=5, results_parent_path=results_parent_path)
        
        ''' n=40 对比 '''
        # n = 40
        # tmax = self.tmin + 3
        
        # all_save_model = True
        # freq_map = self._get_freqs_map(n)
        
        # results_parent_path = current_cache_data_path / f"test_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF_e_GRID"
        # self.ANBF_e_GRID(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path, save_model=all_save_model)
        
        ''' n=40 热力图 '''
        # n = 40
        # tmax = self.tmin + 3
        
        # freq_map = self._get_freqs_map(n)
        
        # self.ANBF_e_ManualGrid(freq_map, n, tmax)
        
        ''' n=8 对比 '''
        # n = 8
        # tmax = 5.5

        # all_save_model = True
        # freq_map = self._get_freqs_map(n)
        
        # results_parent_path = current_cache_data_path / f"test_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF_e_GRID"
        # self.ANBF_e_GRID(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path, save_model=all_save_model)
        
        # results_parent_path = current_cache_data_path / f"test_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF_GRID"
        # self.FBANBF_GRID(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=4, results_parent_path=results_parent_path, save_model=all_save_model)
        
        '''
        results_parent_path = current_cache_data_path / f"test_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF_e_GRID"
        self.FBANBF_e_GRID(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path, save_model=all_save_model)
        '''
        ''' n=8 热力图 '''
        # n = 8
        # tmax = 5.5
        
        # freq_map = self._get_freqs_map(n)
        
        # self.ANBF_e_ManualGrid(freq_map, n, tmax)
        # self.FBANBF_ManualGrid(freq_map, n, tmax)
        
        ''' n=8 固定GRID参数 5/3 s'''
        n = 8

        # tmax = 5.5
        # tmax = self.tmin + 3

        # freq_map = self._get_freqs_map(n)

        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF"
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=12, results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF_e"
        # self.ANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=7, main_weight=1.0, lr_weight=0.4, results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF"
        # self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=3, n_banks=3, band_weights=[1, 1.6, 0.4], results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF_e"
        # self.FBANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=3, n_banks=3, band_weights=[1, 1.6, 0.4], main_weight=1.0, lr_weight=0.4, results_parent_path=results_parent_path)
        

        ''' n=40 固定GRID参数 3 s'''
        # n = 40
        # tmax = self.tmin + 3
        # freq_map = self._get_freqs_map(n)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF_e"
        # self.ANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=4, main_weight=1.0, lr_weight=0.4, results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF"
        # self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, n_banks=3, band_weights=[1, 1.6, 0.4], results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF_e"
        # self.FBANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, n_banks=3, band_weights=[1, 1.6, 0.4], main_weight=1.0, lr_weight=0.4, results_parent_path=results_parent_path)
                
        ''' n=40 固定GRID参数 1 s'''
        # n = 40
        # tmax = self.tmin + 1
        # freq_map = self._get_freqs_map(n)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF"
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=2, results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "ANBF_e"
        # self.ANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=4, main_weight=1.0, lr_weight=0.4, results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF"
        # self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, n_banks=3, band_weights=[1, 1.6, 0.4], results_parent_path=results_parent_path)
        
        # results_parent_path = current_cache_data_path / f"all_anbf_fixed_grid_params_tmax_n{n}" / f"tmax{tmax}_n{n}" / "FBANBF_e"
        # self.FBANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, n_banks=3, band_weights=[1, 1.6, 0.4], main_weight=1.0, lr_weight=0.4, results_parent_path=results_parent_path)
                
        ''' Test ANBF Type 
        # n = 8
        # tmax = 5.5
        # freq_map = self._get_freqs_map(n)
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1)
        # self.ANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, main_weight=1.8, lr_weight=0.8)
        # self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1, n_banks=3, band_weights=[1, 1.6, 0.2])
        # self.FBANBF_e_GRID(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1, n_banks=3, band_weights=[1, 1.6, 0.2])
        
        # n = 40
        # tmax = 5.5
        # freq_map = self._get_freqs_map(n)
        # self.ANBF(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1)
        # self.ANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=True, n_job=1, main_weight=0.9, lr_weight=0.6)
        '''
        
    # Grid algorithm
    def ANBF_e_GRID(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, save_model=False):
        method_name = inspect.currentframe().f_code.co_name
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "main_weight:", "grid", "lr_weight:", "grid")
        pipelines = {}
        
        anbf_filters, anbf_map = self._generate_anbf_filters(list(freq_map.values()), [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        pipelines[method_name] = make_pipeline(SSVEP_ANBF_e(freq_map, anbf_map))
        
        evalution = CrossSubjectEvaluation(
            datasets=self.datasets,
            paradigm=paradigm, 
            overwrite=overwrite, 
            n_jobs=n_job, 
            hdf5_path=results_parent_path, 
            return_epochs=False,
            save_model=save_model,
            verbose=False
        )
        
        param_grid = {}
        param_grid["ANBF_e_GRID"] = {
            "ssvep_anbf_e__main_weight": np.arange(0.2, 2.0 + 0.1, 0.2).tolist(), 
            "ssvep_anbf_e__lr_weight": np.arange(0.2, 2.0 + 0.1, 0.2).tolist()
        }
        
        results = evalution.process(pipelines=pipelines, param_grid=param_grid)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    def FBANBF_GRID(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, save_model=False):
        n_banks = 3
        
        method_name = inspect.currentframe().f_code.co_name        
        pipelines = {}
        ''''''
        freqs = np.array(list(freq_map.values()))
        offset = np.array([i for i in range(1, n_banks + 1)])
        extended_freqs = (freqs[:, None] * offset).ravel()
        ''''''
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "n_banks:", n_banks, "band_weights:", "grid", "extended_freqs:", extended_freqs)
        
        anbf_filters, anbf_map = self._generate_anbf_filters(extended_freqs)
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        pipelines[method_name] = make_pipeline(SSVEP_FBANBF(n_banks, freq_map, anbf_map))
        
        evalution = CrossSubjectEvaluation(
            datasets=self.datasets,
            paradigm=paradigm, 
            overwrite=overwrite, 
            n_jobs=n_job, 
            hdf5_path=results_parent_path, 
            return_epochs=False,
            save_model=save_model,
            verbose=False
        )

        param_grid = {}
        param_grid["FBANBF_GRID"] = {
            "ssvep_fbanbf__band_weights": [[1, i_2, i_3] for i_2 in np.arange(0.0, 2.0 + 0.1, 0.2).tolist() for i_3 in np.arange(0.0, 1.0 + 0.1, 0.2).tolist()]
            }
        
        results = evalution.process(pipelines=pipelines, param_grid=param_grid)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    def FBANBF_e_GRID(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, save_model=False):
        n_banks = 3
        method_name = inspect.currentframe().f_code.co_name        
        pipelines = {}
        ''''''
        freqs = np.array(list(freq_map.values()))
        offset = np.array([i for i in range(1, n_banks + 1)])
        extended_freqs = (freqs[:, None] * offset).ravel()
        ''''''
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "extended_freqs:", extended_freqs, "n_banks:", "grid", "band_weights:", "grid", "main_weight:", "grid", "lr_weight:", "grid")
        
        anbf_filters, anbf_map = self._generate_anbf_filters(extended_freqs, [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        pipelines[method_name] = make_pipeline(SSVEP_FBANBF_e(n_banks, freq_map, anbf_map))
        
        evalution = CrossSubjectEvaluation(
            datasets=self.datasets,
            paradigm=paradigm, 
            overwrite=overwrite, 
            n_jobs=n_job, 
            hdf5_path=results_parent_path, 
            return_epochs=False,
            save_model=save_model,
            verbose=False
        )

        param_grid = {}
        param_grid["FBANBF_e_GRID"] = {
            "ssvep_fbanbf_e__band_weights": [[1, i_2, i_3] for i_2 in np.arange(0.2, 2.0 + 0.1, 0.2).tolist() for i_3 in np.arange(0.2, 1.0 + 0.1, 0.2).tolist()],
            "ssvep_fbanbf_e__main_weight": np.arange(0.2, 2.0 + 0.1, 0.2).tolist(),
            "ssvep_fbanbf_e__lr_weight": np.arange(0.2, 2.0 + 0.1, 0.2).tolist()
            }
        
        results = evalution.process(pipelines=pipelines, param_grid=param_grid)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean() 

    def ANBF_e_ManualGrid(self, freq_map, n_class, tmax):
        def _anbf_e_grid(params, tmax, n, freq_map):
            main_weight = params['main_weight']
            lr_weight = params['lr_weight']
            results_parent_path = current_cache_data_path / f"anbf_e_grid_m_lr_n{n}_tmax{tmax}" / f"m{main_weight:.2f}_lr{lr_weight:.2f}"
            score = self.ANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path, main_weight=main_weight, lr_weight=lr_weight)
            return params, score
        
        param_grid = {
            'main_weight': np.arange(0.2, 2.0 + 0.1, 0.2).tolist(),
            'lr_weight': np.arange(0.2, 2.0 + 0.1, 0.2).tolist()
        }
        results = Parallel(n_jobs=2, backend='threading')(
            delayed(_anbf_e_grid)(params, tmax, n_class, freq_map) for params in ParameterGrid(param_grid)
        )
        best_params, best_score = max(results, key=lambda x: x[1])
        print(best_params, best_score)

    def FBANBF_ManualGrid(self, freq_map, n_class, tmax):  
        def _fbanbf_grid(params, tmax, n, freq_map, n_banks):
            second_weight = params['second_weight']
            third_weight = params['third_weight']
            results_parent_path = current_cache_data_path / f"fbanbf_grid_secondW_thirdW_n{n}_tmax{tmax}" / f"secondW{second_weight:.2f}_thirdW{third_weight:.2f}"
            score = self.FBANBF(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path, n_banks=n_banks, band_weights=[1, second_weight, third_weight])
            return params, score
        
        n_banks = 3
        
        param_grid = {
            'second_weight': np.arange(0.2, 2.0 + 0.1, 0.2).tolist(),
            'third_weight': np.arange(0.2, 1.0 + 0.1, 0.2).tolist()
            }
        results = Parallel(n_jobs=2, backend='threading')(
            delayed(_fbanbf_grid)(params, tmax, n_class, freq_map, n_banks) for params in ParameterGrid(param_grid)
        )
        best_params, best_score = max(results, key=lambda x: x[1])
        print(best_params, best_score)

    def FBANBF_e_ManualGrid(self, freq_map, n_class, tmax):
        def _fbanbf_e_grid(params, tmax, n, freq_map, n_banks):
            second_weight = params['second_weight']
            third_weight = params['third_weight']
            main_weight = params['main_weight']
            lr_weight = params['lr_weight']
            results_parent_path = current_cache_data_path / f"fbanbf_e_grid_m_lr_n{n}_tmax{tmax}_nbanks{n_banks}" / f"secondW{second_weight:.2f}_thirdW{third_weight:.2f}_m{main_weight:.2f}_lr{lr_weight:.2f}"
            score = self.FBANBF_e(freq_map, n_classes=n, tmax=tmax, overwrite=False, n_job=1, results_parent_path=results_parent_path, n_banks=n_banks, band_weights=[1, second_weight, third_weight], main_weight=main_weight, lr_weight=lr_weight)
            return params, score
        
        n_banks = 3
        
        param_grid = {
            'second_weight': np.arange(0.2, 2.0 + 0.1, 0.2).tolist(),
            'third_weight': np.arange(0.2, 1.0 + 0.1, 0.2).tolist(),
            'main_weight': np.arange(0.2, 2.0 + 0.1, 0.2).tolist(),
            'lr_weight': np.arange(0.2, 2.0 + 0.1, 0.2).tolist()
        }
        results = Parallel(n_jobs=2, backend='threading')(
            delayed(_fbanbf_e_grid)(params, tmax, n_class, freq_map, n_banks) for params in ParameterGrid(param_grid)
        )
        best_params, best_score = max(results, key=lambda x: x[1])
        print(best_params, best_score)

    # Base algorithm
    def CCA(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None):
        method_name = inspect.currentframe().f_code.co_name
        print(method_name, "tmax:", tmax, "n_classes:", n_classes)
        pipelines = {}

        paradigm = SSVEP(channels=self.selected_channels, n_classes=n_classes, tmin=self.tmin, tmax=tmax)
             
        pipelines[method_name] = make_pipeline(SSVEP_CCA(freq_map=freq_map, n_harmonics=3))
        
        evalution = CrossSubjectEvaluation(datasets=self.datasets, paradigm=paradigm, overwrite=overwrite, n_jobs=n_job, hdf5_path=results_parent_path)
        
        results = evalution.process(pipelines=pipelines)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    def FBCCA(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, n_filters=7, svd=True):
        method_name = inspect.currentframe().f_code.co_name
        print(method_name, "tmax:", tmax, "n_classes:", n_classes)
        pipelines = {}

        paradigm = CustomFilterBankSSVEP(channels=self.selected_channels, n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=[[8*(n+1), 88] for n in range(n_filters)])
             
        pipelines[method_name] = make_pipeline(SSVEP_FBCCA(n_harmonics=3, freq_map=freq_map, n_filters=n_filters, a=1.25, b=0.25, svd=svd))
        
        evalution = CrossSubjectEvaluation(datasets=self.datasets, paradigm=paradigm, overwrite=overwrite, n_jobs=n_job, hdf5_path=results_parent_path)
        
        results = evalution.process(pipelines=pipelines)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    def ANBF(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, PIPELINE_C=None, PIPELINE_P=None):
        method_name =  PIPELINE_C.__name__ if PIPELINE_C else inspect.currentframe().f_code.co_name
        print(method_name, "tmax:", tmax, "n_classes:", n_classes)
        pipelines = {}

        anbf_filters, anbf_map = self._generate_anbf_filters(list(freq_map.values()))
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters, filter_order=8)
        
        if PIPELINE_C:
            pipelines[method_name] = make_pipeline(PIPELINE_C(freq_map, anbf_map, **PIPELINE_P))
        else:
            pipelines[method_name] = make_pipeline(SSVEP_ANBF(freq_map, anbf_map))
        
        evalution = CrossSubjectEvaluation(datasets=self.datasets, paradigm=paradigm, overwrite=overwrite, n_jobs=n_job, hdf5_path=results_parent_path)
        
        results = evalution.process(pipelines=pipelines)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    # Extented algorithm
    def ANBF_e(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, main_weight=1, lr_weight=1):
        method_name = inspect.currentframe().f_code.co_name
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "main_weight:", main_weight, "lr_weight:", lr_weight)
        pipelines = {}
        
        anbf_filters, anbf_map = self._generate_anbf_filters(list(freq_map.values()), [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        pipelines[method_name] = make_pipeline(SSVEP_ANBF_e(freq_map, anbf_map, main_weight=main_weight, lr_weight=lr_weight))
        
        evalution = CrossSubjectEvaluation(datasets=self.datasets, paradigm=paradigm, overwrite=overwrite, n_jobs=n_job, hdf5_path=results_parent_path, return_epochs=False, verbose=False)
        
        results = evalution.process(pipelines=pipelines)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    def FBANBF(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, n_banks=2, band_weights=[1, 1, 1]):
        method_name = inspect.currentframe().f_code.co_name        
        pipelines = {}
        ''''''
        freqs = np.array(list(freq_map.values()))
        offset = np.array([i for i in range(1, n_banks + 1)])
        extended_freqs = (freqs[:, None] * offset).ravel()
        ''''''
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "n_banks:", n_banks, "band_weights:", band_weights, "extended_freqs:", extended_freqs)
        
        anbf_filters, anbf_map = self._generate_anbf_filters(extended_freqs)
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        pipelines[method_name] = make_pipeline(SSVEP_FBANBF(n_banks, freq_map, anbf_map, band_weights=band_weights))
        
        evalution = CrossSubjectEvaluation(datasets=self.datasets, paradigm=paradigm, overwrite=overwrite, n_jobs=n_job, hdf5_path=results_parent_path, return_epochs=False, verbose=False)
        
        results = evalution.process(pipelines=pipelines)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean()

    def FBANBF_e(self, freq_map, n_classes=40, tmax=5.5, overwrite=False, n_job=1, results_parent_path=None, n_banks=2, band_weights=[1, 1, 1], main_weight=1, lr_weight=1):
        method_name = inspect.currentframe().f_code.co_name        
        pipelines = {}
        ''''''
        freqs = np.array(list(freq_map.values()))
        offset = np.array([i for i in range(1, n_banks + 1)])
        extended_freqs = (freqs[:, None] * offset).ravel()
        ''''''
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "extended_freqs:", extended_freqs, "n_banks:", n_banks, "band_weights:", band_weights, "main_weight:", main_weight, "lr_weight:", lr_weight)
        
        anbf_filters, anbf_map = self._generate_anbf_filters(extended_freqs, [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        pipelines[method_name] = make_pipeline(SSVEP_FBANBF_e(n_banks, freq_map, anbf_map, band_weights, main_weight, lr_weight))
        
        evalution = CrossSubjectEvaluation(datasets=self.datasets, paradigm=paradigm, overwrite=overwrite, n_jobs=n_job, hdf5_path=results_parent_path, return_epochs=False, verbose=False)
        
        results = evalution.process(pipelines=pipelines)
        if self.print_review:    
            self._preview(results)
        return results['score'].mean() 

    # others
    def _latex_table(self):
        temp = [{0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0, 4: 14.0, 5: 15.0, 6: 8.0, 7: 9.0},
            {0: 10.0, 1: 10.2, 2: 11.0, 3: 11.2, 4: 12.0, 5: 12.2, 6: 13.0, 7: 13.2, 8: 14.0, 9: 14.2, 10: 15.0, 11: 15.2, 12: 8.0, 13: 8.2, 14: 9.0, 15: 9.2},
            {0: 10.0, 1: 10.2, 2: 10.4, 3: 11.0, 4: 11.2, 5: 11.4, 6: 12.0, 7: 12.2, 8: 12.4, 9: 13.0, 10: 13.2, 11: 13.4, 12: 14.0, 13: 14.2, 14: 14.4, 15: 15.0, 16: 15.2, 17: 15.4, 18: 8.0, 19: 8.2, 20: 8.4, 21: 9.0, 22: 9.2, 23: 9.4},
            {0: 10.0, 1: 10.2, 2: 10.4, 3: 10.6, 4: 11.0, 5: 11.2, 6: 11.4, 7: 11.6, 8: 12.0, 9: 12.2, 10: 12.4, 11: 12.6, 12: 13.0, 13: 13.2, 14: 13.4, 15: 13.6, 16: 14.0, 17: 14.2, 18: 14.4, 19: 14.6, 20: 15.0, 21: 15.2, 22: 15.4, 23: 15.6, 24: 8.0, 25: 8.2, 26: 8.4, 27: 8.6, 28: 9.0, 29: 9.2, 30: 9.4, 31: 9.6},
            {0: 10.0, 1: 10.2, 2: 10.4, 3: 10.6, 4: 10.8, 5: 11.0, 6: 11.2, 7: 11.4, 8: 11.6, 9: 11.8, 10: 12.0, 11: 12.2, 12: 12.4, 13: 12.6, 14: 12.8, 15: 13.0, 16: 13.2, 17: 13.4, 18: 13.6, 19: 13.8, 20: 14.0, 21: 14.2, 22: 14.4, 23: 14.6, 24: 14.8, 25: 15.0, 26: 15.2, 27: 15.4, 28: 15.6, 29: 15.8, 30: 8.0, 31: 8.2, 32: 8.4, 33: 8.6, 34: 8.8, 35: 9.0, 36: 9.2, 37: 9.4, 38: 9.6, 39: 9.8}
            ]
        
        temp_reverse = []
        
        for original in temp:
            reverse = {str(v): k for k, v in original.items()}
            temp_reverse.append(reverse)
        
        for row_i in range(40):
            freq_map = temp[4]
            value = freq_map[row_i]
            
            for col in range(5):
                if str(value) in temp_reverse[col]:
                    print(f"{temp_reverse[col][str(value)]+1} & {value}", end=" & ")
                else:
                    print("& ", end=" & ")
            print()

    def _preview(self, results):
        mean_score = results['score'].mean()
        print(results.head())
        print(f"Size: {results.shape}, Accuracy: {mean_score:.4f}")
        return mean_score

    def _generate_anbf_filters(self, freqs_list, custom_offsets=[-0.2, -0.1, 0.0, 0.1, 0.2]):
        freq_vals = np.array(freqs_list)
        offsets = np.array(custom_offsets)
        overlap_subfreqs = (freq_vals[:, None] + offsets).ravel()                       # Convert to 1D array
        
        filters_subfreqs = np.unique(np.round(overlap_subfreqs, 1))                     # array([ 7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,
        filters:list[list[float]] = np.column_stack((filters_subfreqs - 0.1, filters_subfreqs + 0.1))     # List of list [[7.8-0.1, 7.8+0.1], ]
        
        single_subfreq_to_index:dict[float, int] = {val: i for i, val  in enumerate(filters_subfreqs)}
        return filters, single_subfreq_to_index

    def _get_freqs_map(self, n_classes) -> dict[int, float]:
        dataset = self.datasets[0]
        origin_subject_list = dataset.subject_list.copy()
        dataset.subject_list = dataset.subject_list[:1]    
        """
        events_id = paradigm.used_events(dataset=dataset)
        print(events_id)
        clf = SSVEP_CCA(n_harmonics=3)
        clf.fit(X, y)
        inferred_freqs = set(np.round(list(clf.class_freqs_.values()), 8))
        print(inferred_freqs)
        """
        paradigm = SSVEP(channels=self.selected_channels, n_classes=n_classes)

        X, y, metadata = paradigm.get_data(dataset=dataset)    
        le = LabelEncoder()
        le.fit_transform(y)
        freq_map = {}
        
        for i, label_str in enumerate(le.classes_):
            freq = float(label_str)
            freq_map[i] = freq
        
        dataset.subject_list = origin_subject_list
        print("Freqencies Mapping:", freq_map)
        return freq_map

    def _get_freqs_map_origin(self, n_classes) -> dict[str, float]:
        dataset = self.datasets[0]
        origin_subject_list = dataset.subject_list.copy()
        dataset.subject_list = dataset.subject_list[:1]    

        paradigm = SSVEP(channels=self.selected_channels, n_classes=n_classes)

        X, y, metadata = paradigm.get_data(dataset=dataset) 
        
        le = LabelEncoder()
        le.fit_transform(y)
        freq_map = {}
        
        for i, label_str in enumerate(le.classes_):
            freq = float(label_str)
            freq_map[label_str] = freq
        
        dataset.subject_list = origin_subject_list
        print("Freqencies Mapping:", freq_map)
        return freq_map

    ''' Not suitable
    def ANBF_e_NestedCV_with_results(self, freq_map, n_classes=40, tmax=5.5, output_dir=None):
        """
        手动实现嵌套交叉验证，返回与 evalution.process 相同格式的 results DataFrame。
        实时保存结果到文件，防止数据丢失。
        """
        if output_dir is None:
            output_dir = Path.cwd() / "nested_cv_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "nested_cv_scores.csv"
        params_file = output_dir / "nested_cv_best_params.csv"
        
        subjects = self.datasets[0].subject_list
        # 预生成滤波器
        anbf_filters, anbf_map = self._generate_anbf_filters(list(freq_map.values()), 
                                                            [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        
        paradigm = CustomFilterBankSSVEP_ANBF(
            channels=['O1-O2'],
            n_classes=n_classes,
            tmin=self.tmin,
            tmax=tmax,
            filters=anbf_filters
        )
        
        X_all, y_all, meta_all = paradigm.get_data(
            self.datasets[0], 
            subjects=subjects,
            cache_config={"use": False}
        )
        
        meta_all = meta_all.reset_index(drop=True)
        subject_to_idx = {}
        for subj in subjects:
            idx = meta_all[meta_all['subject'] == subj].index.values
            subject_to_idx[subj] = idx
        
        all_records = []  # 可选，内存中保留所有记录
        
        # 检查已完成的 subject（断点续传）
        completed_subjects = set()
        if results_file.exists():
            existing_df = pd.read_csv(results_file)
            if 'subject' in existing_df.columns:
                completed_subjects = set(existing_df['subject'].values)
            print(f"Found {len(completed_subjects)} already completed subjects: {completed_subjects}")
        
        for test_subj in subjects:
            if test_subj in completed_subjects:
                print(f"\n=== Skipping already completed subject {test_subj} ===")
                # 仍需将已有记录加载到 all_records 中（可选）
                continue
            
            print(f"\n=== Testing on subject {test_subj} ===")
            
            test_idx = subject_to_idx[test_subj]
            train_idx = np.concatenate([subject_to_idx[s] for s in subjects if s != test_subj])
            
            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]
            
            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            
            # 内层网格搜索
            base_pipeline = make_pipeline(SSVEP_ANBF_e(freq_map, anbf_map))
            param_grid = {
                'ssvep_anbf_e__lr_weight': np.arange(0.2, 2.0, 0.2).tolist(),
                'ssvep_anbf_e__main_weight': np.arange(0.2, 2.0, 0.2).tolist(),
            }
            grid_search = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=5,
                scoring='accuracy',
                refit=True,
                n_jobs=1,
                verbose=0
            )
            
            start_time = time_module.time()
            grid_search.fit(X_train, y_train)
            fit_time = time_module.time() - start_time
            
            score = grid_search.score(X_test, y_test)
            best_params = grid_search.best_params_
            best_inner_score = grid_search.best_score_
            
            # 构造记录
            record = {
                'score': score,
                'time': fit_time,
                'samples': len(X_train),
                'samples_test': len(X_test),
                'n_classes': n_classes,
                'subject': test_subj,
                'session': 'test',
                'channels': len(self.selected_channels),
                'n_sessions': 1,
                'dataset': self.datasets[0].code,
                'pipeline': 'ANBF_e_GRID'
            }
            all_records.append(record)
            
            # 实时写入 scores CSV（追加模式）
            pd.DataFrame([record]).to_csv(results_file, mode='a', header=not results_file.exists(), index=False)
            
            # 保存最佳超参数
            param_record = {
                'subject': test_subj,
                'best_score': score,
                'best_inner_score': best_inner_score,
                'best_params_main_weight': best_params.get('ssvep_anbf_e__main_weight'),
                'best_params_lr_weight': best_params.get('ssvep_anbf_e__lr_weight'),
                'fit_time': fit_time
            }
            pd.DataFrame([param_record]).to_csv(params_file, mode='a', header=not params_file.exists(), index=False)
            
            print(f"Subject {test_subj} test score = {score:.4f}, best inner CV = {best_inner_score:.4f}")
            print(f"Best params: {best_params}")
            
            # 释放内存
            del grid_search
            gc.collect()
        
        # 最终读取所有保存的结果，返回 DataFrame
        if results_file.exists():
            results_df = pd.read_csv(results_file)
            print(f"\nNested CV mean score = {results_df['score'].mean():.4f}")
        else:
            results_df = pd.DataFrame()
            print("No results saved.")
        
        return results_df

    def ANBF_e_StandardGrid(self, freq_map, n_classes=40, tmax=5.5, export_path=None):
        method_name = inspect.currentframe().f_code.co_name
        print(method_name, "tmax:", tmax, "n_classes:", n_classes, "main_weight:", "grid", "lr_weight:", "grid")

        anbf_filters, anbf_map = self._generate_anbf_filters(list(freq_map.values()), [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        paradigm = CustomFilterBankSSVEP_ANBF(channels=['O1-O2'], n_classes=n_classes, tmin=self.tmin, tmax=tmax, filters=anbf_filters)
        
        X_all, y_all, metadata_all = paradigm.get_data(self.datasets[0])
        
        base_pipeline = make_pipeline(SSVEP_ANBF_e(freq_map, anbf_map))
        
        param_grid = {
                'ssvep_anbf_e__main_weight': np.arange(0.2, 2.0, 0.2).tolist(),
                'ssvep_anbf_e__lr_weight': np.arange(0.2, 2.0, 0.2).tolist(),
        }
        
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=5,                     # 内层交叉验证，不再有外层留出
            scoring='accuracy',       # 根据你的指标
            refit=True,               # 找到最佳参数后用全量数据重新训练
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_all, y_all)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        cv_results = pd.DataFrame(grid_search.cv_results_)
        
        print(f"Best params: {best_params}")
        print(f"Best CV score: {best_score:.4f}")
        
        # 导出结果到文件
        if export_path is None:
            export_path = Path.cwd() / f"{method_name}_results"
        else:
            export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # 保存最佳参数和分数
        best_summary = pd.DataFrame([{
            'best_params': str(best_params),
            'best_score': best_score,
            'param_main_weight': best_params['ssvep_anbf_e__main_weight'],
            'param_lr_weight': best_params['ssvep_anbf_e__lr_weight']
        }])
        best_summary.to_csv(export_path / "best_params.csv", index=False)
        
        # 保存全部 cv_results_
        cv_results.to_csv(export_path / "grid_search_cv_results.csv", index=False)
        
        print(f"Results saved to {export_path}")
        pass
    '''

if __name__ == "__main__":
    SSVEP_Test().main()
    