import pandas as pd
import h5py
import os
from .run_spec import RunSpec

# Example:
# results_.hdf5 path: 
# cache_data_064\all_anbf_fixed_grid_params_tmax_n40\tmax3.64_n40\ANBF_e\results\CustomFilterBankSSVEP_ANBF\CrossSubjectEvaluation\results_.hdf5
# dataPath = "cache_data_064\"
# runs = "all_anbf_fixed_grid_params_tmax_n40\tmax3.64_n40\ANBF_e\"
# paradigm = "CustomFilterBankSSVEP_ANBF"

def load_folder(data_path, folder: str, method: str, paradigm: str, tmax: float = None):
    runs = []
    for item in os.listdir(data_path / folder):
        run = RunSpec(folder, item, method, tmax)
        runs.append(run)
    return load_all(data_path, runs, paradigm)

def load_all(data_path, runs: list[RunSpec], paradigm: str):
    dfs = []
    for spec in runs:        
        df = load_one(data_path, spec, paradigm)
        dfs.append(df)
        
    return pd.concat(dfs, axis=0, ignore_index=True)

def load_one(data_path, spec: RunSpec, paradigm: str):
    fullPath = data_path / spec.results_folder / spec.results_parent
    df = load(
            fullPath, 
            paradigm, 
            addtional_info = {
                "Folder Name": spec.results_folder, 
                "Method": spec.method, 
                "Window length (s)": spec.window_length,
                "w_c": spec.w_c,
                "w_lr": spec.w_lr,
                "w_2": spec.w_2,
                "w_3": spec.w_3
            },
            select_index = spec.index
            )
    return df
    
def load(results_parent, paradigm, evaluation="CrossSubjectEvaluation", hdf5="results_.hdf5", addtional_info:dict=None, select_index=0):
    hdf5_path = results_parent / "results" / paradigm / evaluation / hdf5
    df = loadHDF5(hdf5_path, addtional_info=addtional_info, select_index=select_index)
    return df

def loadHDF5(file_path, addtional_info=None, select_index=0):
        inner_dfs = []
        with h5py.File(file_path, 'r') as f:
            print(list(f.keys()))
            for index, top_key in enumerate(list(f.keys())[select_index:select_index+1]):
                # top_key = list(f.keys())[0]  # 'e404bc1ab5483b39b66561770215631e'
                if select_index != 0:
                    print("select_index:", select_index)
                dataset_group = f[top_key]['Wang2016']
                
                # print(dataset_group.keys())

                data = dataset_group['data'][:]   # 假设这也是一个 numpy 数组
                ids  = dataset_group['id'][:]       # 可能是被试 ID 或其他标签
                
                # print("data shape:", data.shape)
                # print("id shape:", ids.shape)
                
                subject_ids = [id_pair[0].decode('utf-8') for id_pair in ids]
                session_ids = [id_pair[1].decode('utf-8') for id_pair in ids]

                df = pd.DataFrame(data, columns=['Accuracy', 'time', 'samples', 'samples_test', 'n_classes'])
                df['subject'] = subject_ids
                df['session'] = session_ids
                
                if addtional_info:
                    for k, v in addtional_info.items():
                        df[k] = v
                
                df['file'] = file_path.name
                # df['file_path'] = file_path
                
                # print(df.head())
                inner_dfs.append(df)
                
        return pd.concat(inner_dfs, axis=0, ignore_index=True)
    
