# AMORe: An Agnostic Multi-Objective Framework for Recommendation
This repository contains the source codes and datasets of the paper _AMORe: An Agnostic Multi-Objective Framework for Recommendation_.

### Requirements
We implemented and tested the models in Python `3.8.10`, with `PyTorch==2.0.1` and CUDA `11.7`. The `NGCF` model require `PyTorch Geometric`. Then, the requirements listed in the `requirements.txt` file refer to these versions. You may create the virtual environment with the requirements file as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Data
In the folder `data`, you can find the data used in our work (`Amazon Baby`, `Facebook Books`, `Amazon Music`). We provide the split version of the data.

### Run the models
In the following, we explain how to run the models within the paper.
- BPRMF, NGCF, MultiFR, and AMORe can be executed through the `main.py` script. Specifically, you should refer to the configuration files contained into the folder `config_files`. You may train the model by running the following code:
  ```
  $ CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main.py --config [CONFIGURATION_FILE_NAME]
  ```
  The configuration files are in the folder `config_files`. Consider the following schema:
  - `_dataset_backbone_none.yml` as the configuration files for vanilla backbone by changing `dataset` and `backbone` at you convenience. 
  - `_dataset_backbone_AMORE_SCALE_cp.yml` as the configuration files for AMORe with static scalarization by changing `dataset` and `backbone` at you convenience. 
  - `_dataset_backbone_AMORE_MGDA_cp.yml` as the configuration files for AMORe with MGDA by changing `dataset` and `backbone` at you convenience. 
  - `_dataset_backbone_AMORE_EPO_cp.yml` as the configuration files for AMORe with EPO by changing `dataset` and `backbone` at you convenience. 
  - `_dataset_backbone_multifr_cp.yml` as the configuration files for MultiFR.
- CPFAIR can be executed through the `main_cpfair.py` script. It requires the matrix of scores (user, item) predicted by a backbone model (in this paper BPRMF and NGCF). For this reason, this code is prepared in order to load such matrix saved with `.npz` [extension](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) from a folder called `arrays`.  You may train the model by running the following code:
  ```
  $ python3 -u main_cpfair.py
  ```
  Please, note that to execute this baseline, you need to install the Gurobi package on your machine.

### Evaluation
To evaluate the models, we relied on the public and open-source framework Elliot. Here, given a pre-obtained recommendation list by running the models as explained above, you can compute the metrics discussed within the paper. Please, refer to Elliot official [documentation](https://elliot.readthedocs.io/en/latest/) for further details on how compute the metrics. We add an example of configuration file in the `config_files` folder within the compressed file.

