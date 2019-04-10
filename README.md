# distinct_events _asc
Acoustic Scene Classification by Implicitly Identifying Distinct Sound Events

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
python -m ipykernel install --user --name asc_mt --display-name 'python3.6(asc_mt)'
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
n_fft = 1102
hop_length = 1102
n_mels = 64
```

### extract and store feature in .h5 file
```
# generate .h5 files under data_manager/data_h5 
python data_manager/dcase18_taskb.py
# generate scaler .h5 under data_manager/data_h5
python data_manager/taskb_standrizer.py
```

## model
