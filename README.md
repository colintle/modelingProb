# Weather Impacts and Outage Prediction Using Distribution Networks' Topology and Physical Features - Outage Model CLI

This project includes a Command Line Interface (CLI) for preprocessing weather and static data for model training, as well as training and validating a GATRNN model for predicting outages.

## Authors

- Colin T. Le
- Kenneth McDonald
- Zhihua Qu

## Setup

To get the CLI running, first install the package in editable mode:

```bash
pip install --editable .
```

Then, visit the [PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to install the correct PyTorch package for either CPU or CUDA.

## CLI Commands

### 1. Preprocess Data
```
outage-model preprocess [OPTIONS]
```
Preprocess the weather and static data for model training.

**Options:**
- `--node-static-features`: One or more physical node features to be considered for modeling.  
  Example: `--node-static-feature elevation`
- `--edge-static-feature`: One or more physical edge features to be considered for modeling.  
  Example: `--edge-static-feature length`
- `--data-folder PATH`: Input relative path to the folder containing the model data.
- `--weather-features`: One or more weather features (see the list of possible weather events).
- `--output PATH`: Output path to save both the CSV file and the pickle file.

### 2. Train and Validate Model
```
outage-model train-validate [OPTIONS]
```
Train and validate the GATRNN model with the given parameters.

**Options:**
- `--pkl-file PATH`: Input path to the pickle file containing datasets.
- `--epochs INT`: Number of training epochs.
- `--learning-rate FLOAT`: Learning rate for the optimizer.
- `--hidden-size INT`: Hidden size for the model.
- `--validation-scale FLOAT`: Scale for validation data.
- `--output-model-file PATH`: Output path to save the trained model as a `.pth` file.

## Example Commands
Below are example commands for two types of datasets: Real Weather and Synthetic Weather.

Before Running These Commands

1. Download the datasets from the links provided (not shown here).
2. Create a new folder called ./model_data in the root directory of this project.
3. Place the downloaded .hdf5 files in the ./model_data folder.

### Real Weather Dataset

1. Preprocess Data

```
outage-model preprocess \
    --dataset-file model_data/datasets_real_weather.hdf5 \
    --edge-static-features vegetation \
    --edge-static-features length \
    --node-static-features elevation \
    --node-static-features vegetation \
    --node-static-features flood_zone_num \
    --node-static-features Depth \
    --weather-features wspd \
    --weather-features prcp \
    --output output_real
```

2. Train and Validate

```
outage-model train-validate \
    --pkl-file output_real/dataDict.pkl \
    --output-model-file output_real/gatrnn_model.pth \
    --epochs 1500 \
    --learning-rate 0.001 \
    --hidden-size 40 \
    --optimizer adam
```

3. Evaluate

```
outage-model evaluate-gatrnn \
    --pkl-file output_real/dataDict.pkl \
    --model-file output_real/gatrnn_model.pth
```

### Synthetic Weather Dataset

1. Preprocess Dataset

```
outage-model preprocess \
    --dataset-file model_data/datasets_fake_weather.hdf5 \
    --edge-static-features vegetation \
    --edge-static-features length \
    --node-static-features elevation \
    --node-static-features vegetation \
    --node-static-features flood_zone_num \
    --node-static-features Depth \
    --weather-features wspd \
    --weather-features prcp \
    --output output_fake
```

2. Train and Validate

```
outage-model train-validate \
    --pkl-file output_fake/dataDict.pkl \
    --output-model-file output_fake/gatrnn_model.pth \
    --epochs 100 \
    --learning-rate 0.001 \
    --hidden-size 40 \
    --optimizer adam
```

3. Evaluate

```
outage-model evaluate-gatrnn \
    --pkl-file output_fake/dataDict.pkl \
    --model-file output_fake/gatrnn_model.pth
```
