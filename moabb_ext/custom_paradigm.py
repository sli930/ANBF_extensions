""" Paradigm Classes """
import os
import pickle
import hashlib
import numpy as np
import pandas as pd
import mne
import threading
from typing import Literal, override
from mne import EpochsArray
from concurrent.futures import ThreadPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from moabb.datasets.preprocessing import NamedFunctionTransformer
from operator import methodcaller
from moabb.paradigms import FilterBankSSVEP
from moabb.datasets.bids_interface import StepType
from sklearn.pipeline import Pipeline
import config

cpu_count = os.cpu_count()
current_cache_data_path = config.current_cache_data_path
current_cache_data_path.mkdir(parents=True, exist_ok=True)

class CustomFilterBankSSVEP(FilterBankSSVEP):
    _data_cache = {}
    _cache_lock = threading.Lock()
    
    @override
    def get_data(  # noqa: C901
        self,
        dataset,
        subjects=None,
        return_epochs=False,
        return_raws=False,
        cache_config=None,
        postprocess_pipeline=None,
        process_pipelines=None,
        additional_metadata: Literal["all"] | list[str] = None,
    ):
        if process_pipelines is not None:
            assert isinstance(process_pipelines, list)
            assert isinstance(process_pipelines[0], Pipeline)
            output_step_type, _ = process_pipelines[0].steps[-1]
            if (
                (output_step_type == StepType.ARRAY and (return_epochs or return_raws))
                or (output_step_type == StepType.EPOCHS and not return_epochs)
                or (output_step_type == StepType.RAW and not return_raws)
            ):
                raise ValueError(
                    f"process_pipeline output step type {output_step_type} incompatible with "
                    f"arguments {return_epochs=} and {return_raws=}."
                )

        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        if subjects is None:
            subjects = dataset.subject_list

        # if process_pipelines is None:     
        # # This if-statement is causing the Custom Paradigm Filter functions to fail. 
        # # Because mne\Lib\site-packages\moabb\evaluations\base.py BaseEvaluation._load_data.requires_epochs depends on whether class is a SSVEP_CCA, SSVEP_TRCA or SSVEP_MsetCCA. <- hard-code.
        process_pipelines = self.make_process_pipelines(
            dataset, return_epochs, return_raws, postprocess_pipeline
        )

        labels_pipeline = self.make_labels_pipeline(dataset, return_epochs, return_raws)

        def fetch_data(process_pipeline):
            return dataset.get_data(
                subjects=subjects,
                cache_config=cache_config,
                process_pipeline=process_pipeline,
            )
            
        filter_repr = repr(self.filters) + repr(int(self.n_classes)) + repr(float(self.tmin)) + repr(float(self.tmax))
        filter_hash = hashlib.md5(filter_repr.encode()).hexdigest()
        cache_file = f"cached_data_{dataset.code}_{filter_hash}.pkl"
        pickle_path = current_cache_data_path / cache_file
        cache_key = str(pickle_path)

        # 线程安全的缓存读取
        with self.__class__._cache_lock:
            if cache_key in self.__class__._data_cache:
                print("Using cached data from memory", filter_hash)
                data = self.__class__._data_cache[cache_key]
            else:
                if pickle_path.exists():
                    print("Loading from disk (first time)", filter_hash)
                    with open(pickle_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    print("Creating cache file", filter_hash)
                    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                        data = list(executor.map(fetch_data, process_pipelines))
                    # 写入临时文件后重命名，保证原子性
                    tmp_path = pickle_path.with_suffix('.tmp')
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    tmp_path.rename(pickle_path)
                # 存入缓存
                self.__class__._data_cache[cache_key] = data
            
        X = []
        labels = []
        metadata = []
        for subject, sessions in data[0].items():
            for session, runs in sessions.items():
                for run in runs.keys():
                    proc = [data_i[subject][session][run] for data_i in data]
                    if additional_metadata:
                        ext_metadata = [
                            dataset.get_additional_metadata(
                                subject=subject, session=session, run=run
                            )
                        ] * len(process_pipelines)

                        if isinstance(additional_metadata, list):
                            ext_metadata = [
                                dm[["session", "subject", "run"] + additional_metadata]
                                for dm in ext_metadata
                            ]
                    else:
                        ext_metadata = [None] * len(process_pipelines)

                    if any(obj is None for obj in proc):
                        # this mean the run did not contain any selected event
                        # go to next
                        assert all(obj is None for obj in proc)  # sanity check
                        continue

                    if return_epochs:
                        assert all(len(proc[0]) == len(p) for p in proc[1:])
                        n = len(proc[0])
                        lbs = labels_pipeline.transform(proc[0])
                        if len(self.filters) == 1:
                            x = proc[0]
                        else:
                            for p in proc:
                                p.set_annotations(None)
                            x = self.addRawToFirstEpoch(proc)                                
                            # x = mne.concatenate_epochs(proc)

                    elif return_raws:
                        assert all(len(proc[0]) == len(p) for p in proc[1:])
                        n = 1
                        lbs = labels_pipeline.transform(
                            proc[0]
                        )  # XXX does it make sense to return labels for raws?
                        x = proc[0] if len(self.filters) == 1 else proc
                    else:  # return array
                        assert all(
                            np.array_equal(proc[0]["X"].shape, p["X"].shape)
                            for p in proc[1:]
                        )
                        assert all(
                            np.array_equal(proc[0]["events"], p["events"])
                            for p in proc[1:]
                        )
                        n = proc[0]["X"].shape[0]
                        events = proc[0]["events"]
                        lbs = labels_pipeline.transform(events)
                        x = (
                            proc[0]["X"]
                            if len(self.filters) == 1
                            else np.array([p["X"] for p in proc]).transpose((1, 2, 0, 3))
                        )

                    met = pd.DataFrame(index=range(n))
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run

                    metadata.append(met)

                    # overwrite if additional is required
                    if additional_metadata:
                        # extend the metadata according to the filters

                        dmeta_ext = (
                            ext_metadata[0].copy()
                            if isinstance(ext_metadata[0], pd.DataFrame)
                            else pd.DataFrame()
                        )
                        metadata[-1] = dmeta_ext

                    if return_epochs:
                        x.metadata = (
                            metadata[-1].copy()
                            # if len(self.filters) == 1
                            # else pd.concat(
                            #     [metadata[-1].copy()] * len(self.filters),
                            #     ignore_index=True,
                            # )
                        )

                    X.append(x)
                    labels.append(lbs)

        metadata = pd.concat(metadata, ignore_index=True)
        labels = np.concatenate(labels)
        if return_epochs:
            for ep in X:
                ep.set_annotations(None)
            X = mne.concatenate_epochs(X)
        elif return_raws:
            pass
        else:
            X = np.concatenate(X, axis=0)
        return X, labels, metadata
    
    def addRawToFirstEpoch(self, epochs_list: list[EpochsArray]):
        '''Directly add every epochs raw data into first epoch, keep original number of event.'''
        first_epoch = epochs_list[0]
        for epoch in epochs_list[1:]:
            first_epoch = self.merge_epochs_by_frequency(first_epoch, epoch)
        return first_epoch

    def merge_epochs_by_frequency(self, epochs_a, epochs_b):
        data_a = epochs_a.get_data()
        data_b = epochs_b.get_data()
        
        combined_data = np.concatenate((data_a, data_b), axis=2)   
        
        merged = mne.EpochsArray(
            combined_data,
            info=epochs_a.info,          # 通道信息（如电极名称、采样率等）
            tmin=epochs_a.tmin,          # 起始时间保持不变
            events=epochs_a.events,      # 事件信息保持不变
            event_id=epochs_a.event_id,  # 事件 ID 保持不变
            verbose=False
        )
                
        return merged

class CustomFilterBankSSVEP_ANBF(CustomFilterBankSSVEP):
    def __init__(self, filter_order=8, **kwargs):
        super().__init__(**kwargs)
        self.filter_order = filter_order
    
    def o1_minus_o2(self, raw):
        if 'O1' not in raw.ch_names or 'O2' not in raw.ch_names:
            raise ValueError("O1 or O2 channel not found")
        
        o1_idx = raw.ch_names.index('O1')
        o2_idx = raw.ch_names.index('O2')
        raw._data[o1_idx] = raw._data[o1_idx] - raw._data[o2_idx]
        
        raw.drop_channels(['O2'])
        raw.rename_channels({'O1': 'O1-O2'})
        return raw 

    @override
    def _get_raw_pipelines(self):
        filter_pipelines = super()._get_raw_pipelines()
        diff_transformer = FunctionTransformer(self.o1_minus_o2, validate=False)
            
        return [make_pipeline(diff_transformer, fp) for fp in filter_pipelines]

class CustomFilterBankSSVEP_ANBF_CustomIIR(CustomFilterBankSSVEP_ANBF):
    @override
    def _get_raw_pipelines(self):
        filter_pipelines = [self.get_filter_pipeline(fmin, fmax) for fmin, fmax in self.filters]
        diff_transformer = FunctionTransformer(self.o1_minus_o2, validate=False)
            
        return [make_pipeline(diff_transformer, fp) for fp in filter_pipelines]

    def get_filter_pipeline(self, fmin, fmax):
        # iir_params=dict(order=5, ftype='butter', output='sos')
        iir_params = {
                    'order': 8,       # 滤波器阶数
                    'rs': 60,         # 阻带最小衰减（dB），这是切比雪夫II型滤波器的关键参数
                    'ftype': 'cheby2',# 指定滤波器类型
                    'output': 'sos'   # 输出格式为二阶节（SOS），提升数值稳定性
                }
        
        return NamedFunctionTransformer(
            func=methodcaller(
                "filter",
                l_freq=fmin,
                h_freq=fmax,
                method="iir",
                picks="data",
                iir_params = iir_params,
                verbose=False,
            ),
            display_name=f"Band Pass Filter Order {self.filter_order} ({fmin}–{fmax} Hz)",
        )
