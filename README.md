# Acoustic Scene Classification by Implicitly Identifying Distinct Sound Events
## Introduction
This is the code for the paper arxiv:1904.05204 ([download](https://arxiv.org/abs/1904.05204))

The paper is submitted to Interspeech 2019.

## install
* create env and activate
```bash
conda create -n distinct_events_asc python=3.6 pip
source activate distinct_events_asc
```
* install pytorch torchvision from pytorch channel
```bash
conda install pytorch torchvision -c pytorch
```
* install requirements with pip
```bash
pip install -r requirements.txt
```
* if want to run jupyter notebook examples, install a `kernelspec` for env
```
conda install jupyter ipykernel
python -m ipykernel install --user --name distinct_events_asc --display-name 'python3.6(distinct_events_asc)'
```

## data_manager
*NOTE: before use, config __data_manager.cfg__ properly*
* create file `data_manager.cfg` under `data_manager/`
* specify dev_path to point to dcase2018 Task1 SubTaskB dataset
``` python
[DEFAULT]

[dcase18_taskb]
dev_path = /PathTo.../dcase2018_baseline/task1/datasets/TUT-urban-acoustic-scenes-2018-mobile-development

[logmel]
sr = 44100
n_fft = 1764
hop_length = 882
n_mels = 40
fmax = 22050
```

* extract and store feature in .h5 file
```
# generate .h5 files under data_manager/data_h5 
python data_manager/dcase18_taskb.py
# generate scaler .h5 under data_manager/data_h5
python data_manager/taskb_standrizer.py
```

## Experiments
* open `jupyter notebook` or `jupyter lab`
* run experiments notebooks under `jupyter_exp/`

## Citation
* BibLatex
```
@online{1904.05204,
Author = {Hongwei Song and Jiqing Han and Shiwen Deng and Zhihao Du},
Title = {Acoustic Scene Classification by Implicitly Identifying Distinct Sound Events},
Year = {2019},
Eprint = {1904.05204},
Eprinttype = {arXiv},
}
```
* Bibtex
```
@misc{1904.05204,
Author = {Hongwei Song and Jiqing Han and Shiwen Deng and Zhihao Du},
Title = {Acoustic Scene Classification by Implicitly Identifying Distinct Sound Events},
Year = {2019},
Eprint = {arXiv:1904.05204},
}
```
