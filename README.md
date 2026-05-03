# Anonymous Code for TALoss

This repository provides the anonymized implementation of **Trend-Aligned Loss (TALoss)** for long-term time-series forecasting.

TALoss is a training objective designed for time-series forecasting. In addition to local numerical prediction errors, it introduces a bounded trend-alignment penalty between adjacent time steps. The code also reports Pearson correlation coefficient (PCC) as an additional evaluation metric to measure the correlation between predicted and ground-truth sequences.

## 1. Environment Setup

Create a Python environment and install the required dependencies.

```bash
conda create -n taloss python=3.11
conda activate taloss
pip install -r requirements.txt
```

If `requirements.txt` is not available, the basic dependencies include:

```bash
pip install numpy pandas scikit-learn matplotlib torch
```

Please install the PyTorch version that matches your CUDA version from the official PyTorch website.

## 2. Dataset Preparation

Please organize the datasets under the `dataset` folder. This repository supports the following long-term forecasting datasets:

```text
ETTh1
ETTh2
ETTm1
ETTm2
Weather
Exchange
Electricity
```

A typical dataset structure is:

```text
./dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── exchange_rate/
│   └── exchange_rate.csv
└── electricity/
    └── electricity.csv
```

A typical project structure is:

```text
.
├── run.py
├── exp/
│   └── exp_long_term_forecasting.py
├── models/
├── data_provider/
├── utils/
├── ./dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
    ├── weather/
        └── weather.csv
    ├── exchange_rate/
        └── exchange_rate.csv
    └── electricity/
        └── electricity.csv
└── README.md
```

## 3. Supported Loss Functions

The code supports the following training losses:

```text
MSE
TALoss
```

Use MSE by setting:

```bash
--loss MSE
```

Use TALoss by setting:

```bash
--loss TALoss
```

The main TALoss hyperparameters are:

```bash
--taloss_lambda 0.3
--taloss_eps 1e-8
```

where `taloss_lambda` controls the weight of the trend-alignment penalty, and `taloss_eps` is used for numerical stability.

## 4. Example: Training with TALoss on ETTh1

The following command trains a forecasting model on the ETTh1 dataset with prediction length 720 using TALoss.

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1 \
  --loss TALoss \
  --taloss_lambda 0.5
```

## 5. Example: Training with MSE

To train the same setting with the standard MSE loss, use:

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1 \
  --loss MSE
```

## 6. Testing a Saved Model

After training, the best checkpoint is saved under:

```text
./checkpoints/
```

To test a saved model, set `--is_training 0` and keep the same experimental configuration:

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1 \
  --loss TALoss \
  --taloss_lambda 0.5
```

## 7. Evaluation Metrics

The code reports the following metrics:

```text
MSE
MAE
PCC
```


The main results are saved under:

```text
./results/
```

The prediction visualizations are saved under:

```text
./test_results/
```

For each experiment, the saved files include:

```text
metrics.npy
metrics.txt
pred.npy
true.npy
```

where `metrics.txt` records the numerical results in a readable format.

## 8. Reproducing Different Prediction Horizons

To evaluate different prediction lengths, modify `--pred_len`. For example:

```bash
--pred_len 96
--pred_len 192
--pred_len 336
--pred_len 720
```

A simple loop can be written as:

```bash
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_${pred_len} \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1 \
  --loss TALoss \
  --taloss_lambda 0.5
done
```
