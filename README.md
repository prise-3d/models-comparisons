# Models comparisons

## Description

Project developed in order to compare models performances using simulation of these models.

## Installation


```bash
git clone --recursive https://github.com/prise-3d/models-comparisons.git
```


```bash
pip install -r requirements.txt
```

## Train model ?


### Precompute the whole expected features
First you need to generate data using thresholds file (file obtained from SIN3D app):

```bash
python processing/generate_all_data_file.py --feature lab --dataset /path/to/folder --output output --thresholds file.csv
```

- `--output`: save automatically output into `data/generated`

### Well compare models
In order to well compare models, you need to set the training and testing zones used for your dataset:

```bash
python processing/generate_selected_zones_file.py --dataset /path/to/folder --n_zones 12 --output file --thresholds file.csv
```

- `--output`: save automatically output into `data/learned_zones`

Each image is cut out into 16 zones, then you need to use the `n_zones` parameter to set you number of zones selected for training part.

The generated output file contains information for each scene about indices used for training and testing sets.

### Generate your dataset

Then, you can generate your dataset:

```bash
python processing/generate_all_data_file.py --data data/generated/output --thresholds file.csv --selected_zones data/learned_zones/file --interval 0,40 --kind svdn --feature lab --output data/datasets/name
```

- `--data`: specify the output data folder path generated when precomputing features.
- `--selected_zones`: the previous output file generated in order to set.
- `--interval`: set the interval to use from feature generated.
- `--kind`: normalization level (svn, svdn, svdne).
- `--output`: save automatically output into `data/datasets`.

### Train your model

You can now use your dataset to train your model:

```bash
python train_model.py --data data/datasets/dataset/dataset --output modelv1 --choice svm_model
```

- `--data`: specify the dataset name (without .train and .test generated extension) obtained from previous script.
- `--output`: save automatically output into `data/saved_models`.

## Simulations


### Obtained model simulation on scene

```bash
python simulation/estimate_thresholds_file.py --model data/saved_models/modelv1.joblib --method lab --interval 0,40 --kind svdn --folder /path/to/scene --save filename.csv --label "Simulate modelv1"
```

- `--folder`: scene folder to simulate on.
- `--save`: filename to use as output simulation results (append simulation results)


### Display and compare scene simulations

```bash
python display/display_estimated_file.py --simulation filename.csv --learned_zones --data/learned_zones/file --scene /path/to/scene --thresholds file.csv
```

