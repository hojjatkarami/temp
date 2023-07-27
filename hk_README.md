# Codebase for "Time-series Generative Adversarial Networks (TimeGAN)"

```
install tf 1.15.0

[https://github.com/TachibanaYoshino/AnimeGANv2/issues/48]
conda install tensorflow-estimator=1.15.1
```
[https://fmorenovr.medium.com/install-conda-and-set-up-a-tensorflow-1-15-cuda-10-0-environment-on-ubuntu-windows-2a18097e6a98


conda install -c conda-forge tensorflow-gpu=1.15
scikit-learn>=0.21.3
conda install matplotlib


]
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper Link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Contact: jsyoon0823@gmail.com

This directory contains implementations of TimeGAN framework for synthetic time-series data generation
using one synthetic dataset and two real-world datasets.

- Sine data: Synthetic
- Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
- Energy data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

To run the pipeline for training and evaluation on TimeGAN framwork, simply run
python3 -m main_timegan.py or see jupyter-notebook tutorial of TimeGAN in tutorial_timegan.ipynb.

Note that any model architecture can be used as the generator and
discriminator model such as RNNs or Transformers.

### Code explanation

(1) data_loading.py

- Transform raw time-series data to preprocessed time-series data (Googld data)
- Generate Sine data

(2) Metrics directory
(a) visualization_metrics.py

- PCA and t-SNE analysis between Original data and Synthetic data
  (b) discriminative_metrics.py
- Use Post-hoc RNN to classify Original data and Synthetic data
  (c) predictive_metrics.py
- Use Post-hoc RNN to predict one-step ahead (last feature)

(3) timegan.py

- Use original time-series data as training set to generater synthetic time-series data

(4) main_timegan.py

- Report discriminative and predictive scores for the dataset and t-SNE and PCA analysis

(5) utils.py

- Some utility functions for metrics and timeGAN.

### Command inputs:

- data_name: sine, stock, or energy
- seq_len: sequence length
- module: gru, lstm, or lstmLN
- hidden_dim: hidden dimensions
- num_layers: number of layers
- iterations: number of training iterations
- batch_size: the number of samples in each batch
- metric_iterations: number of iterations for metric computation

Note that network parameters should be optimized for different datasets.

### Example command

```shell
$ python main_timegan.py --data_name stock --seq_len 24 --module gru --hidden_dim 24 --num_layer 3 --iteration 50000 --batch_size 128 --metric_iteration 10
```

### Outputs

- ori_data: original data
- generated_data: generated synthetic data
- metric_results: discriminative and predictive scores
- visualization: PCA and tSNE analysis
