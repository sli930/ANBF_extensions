# ANBF Extensions for SSVEP-Based BCI

This repository contains the implementation and evaluation code for the original ANBF method and three extensions: ANBF_e, FBANBF, and FBANBF_e.

## Requirements

- Python 3.12+
- `moabb`, `mne`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

Install the virtual environment and the specific version of mne and moabb:

```bash
conda create --channel=conda-forge --strict-channel-priority --name=mne mne
pip install mne==1.12.1
pip install moabb==1.5.0
```
### Optional MOABB patch
After installing the environment, copy `base.py` and `utils.py` from `moabb_patch/evaluations/` into your installed MOABB package at `.../site-packages/moabb/evaluations/`, replacing the existing files. You can locate the MOABB package path with:
```bash
python -c "import moabb; print(moabb.__file__)"
```
This step is optional but recommended for reproducing the detailed results, as it saves per-fold outputs and predictions during grid search.

## Usage

Run the main experiment script:
```bash
python main.py
```
The dataset (Wang2016) is downloaded automatically by MOABB on first run. All results are saved in the `cache_data_064/` by default. This path can be changed in `config.py`. `cache_data_064/` stores intermediate results and can be deleted safely; it will be regenerated on the next run.

## Reproducing Results

To reproduce the main tables and figures, run:
```bash
python make_plots.py
```
Outputs are saved in the root directory.

## Code Structure
- `plotting/`: loads results and generates charts.
- `moabb_ext/`: core implementation of ANBF, ANBF_e, FBANBF, and FBANBF_e.
- `cache_data_064/`: cached data for reuse (can be deleted and regenerated). 

## License

This project is released under the MIT License. See LICENSE for details.

## Citation

If you use this code, please cite the corresponding thesis/paper.

## Contact

For questions, please open an issue.