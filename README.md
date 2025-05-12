# RatioWaveNet

RatioWaveNet is a custom deep learning architecture designed for the classification of EEG signals in Motor Imagery (MI) tasks. Inspired by existing models like EEGNet and ATCNet, RatioWaveNet introduces a lightweight CNN framework enhanced with residual connections, dropout, and optimized temporal feature extraction.

Authors : Giuseppe Bonomo, Marco Siino, Rosario Sorbello

University of Palermo, Italia 

---
In addition to the proposed [**RatioWaveNet**](https://github.com/Bonomo31/RatioWaveNet) model, the repository includes implementations of several other well-known EEG classification architectures in the `models.py` file, which can be used as baselines for comparison with RatioWaveNet. These include:

- **ATCNet**:[paper](https://ieeexplore.ieee.org/document/9852687), [original code](https://github.com/Altaheri/EEG-ATCNet)
- **EEGNet**:[paper](https://arxiv.org/abs/1611.08024), [original code](https://github.com/vlawhern/arl-eegmodels)
- **EEG-TCNet**:[paper](https://arxiv.org/abs/2006.00622), [original code](https://github.com/iis-eth-zurich/eeg-tcnet)
- **MBEEG_SENet**:[paper](https://www.mdpi.com/2075-4418/12/4/995)
- **ShallowConvNet**:[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)

The following table summarizes the classification performance of [**RatioWaveNet**](https://github.com/Bonomo31/RatioWaveNet) and the other reproduced models, based on the experimental setup defined in the notebook for each model and dataset.


| Model           | #params | BCI 4-2a Accuracy | BCI 4-2a Kappa | BCI 4-2b Accuracy | BCI 4-2b Kappa | HGD Accuracy | HGD Kappa |
|-----------------|---------|-------------------|----------------|-------------------|----------------|--------------|-----------|
| RatioWaveNet        | 113,732 | 81.56             | 75.40          | 97.69             | 80.60          | 92.81        | 90.40     |
| ATCNet          | 113,732 | 81.10             | 79.73          | 89.41             | 78.80          | 92.05        | 89.40     |
| EEGTCNet        | 4,096   | 77.97             | 70.63          | 83.69             | 67.31          | 87.80        | 83.73     |
| MBEEG_SENet     | 10,170  | 79.98             | 73.30          | 86.53             | 73.02          | 90.13        | 86.84     |
| ShallowConvNet  | 47,310  | 80.52             | 74.02          | 86.02             | 72.38          | 87.00        | 82.67     |
| EEGNet          | 2,548   | 77.68             | 70.24          | 86.08             | 72.13          | 88.25        | 84.33     |

### **Disclaimer**   
> - The results reported for**BCI 4-2a**, **BCI 4-2b** and **HGD** datasets were not recomputed by us and are directly extracted from the original papers.  
> - **HGD (High Gamma Dataset)**: Refers to physically executed movements (executed movements), not motor imagery (motor imagery).


----
# Comparative preprocessing  

The following table presents a comparative analysis of different deep learning models with and without the application of the RDWT (Redundant Discrete Wavelet Transform) preprocessing technique. The evaluation covers three benchmark EEG motor imagery datasets: BCI Competition IV-2a, BCI Competition IV-2b, and the High-Gamma Dataset (HGD). The aim is to assess the impact of RDWT on classification performance (accuracy and Cohen’s kappa score).

In particular, the results highlight the performance improvements achieved by [**RatioWaveNet**](https://github.com/Bonomo31/RatioWaveNet) when combined with the RDWT preprocessing, compared to both its baseline (no preprocessing) and other well-established architectures.


| Model           | Preprocessing | BCI 2a Acc. | BCI 2a κ | BCI 2b Acc. | BCI 2b κ | HGD Acc. | HGD κ |
|----------------|---------------|-------------|----------|-------------|----------|----------|--------|
| RatioWaveNet       | None          | 79.36       | 72.50    | 97.00       | 69.80    | 87.45    | 83.30  |
|                | RDWT          | 81.56       | 75.40    | 97.69       | 80.60    | 92.81    | 90.40  |
| ATCNet         | None          | 79.71       | 72.90    | 96.90       | 63.30    | 88.88    | 85.20  |
|                | RDWT          | 79.51       | 72.70    | 96.74       | 61.90    | 88.26    | 84.30  |
| EEGTCNet       | None          | 64.35       | 52.50    | 95.81       | 58.90    | 86.60    | 82.10  |
|                | RDWT          | 68.79       | 58.40    | 96.09       | 66.60    | 87.14    | 82.90  |
| MBEEG_SENet    | None          | 70.49       | 60.60    | 96.95       | 73.80    | 90.58    | 87.40  |
|                | RDWT          | 72.72       | 63.60    | 96.28       | 63.50    | 90.26    | 87.00  |         
| ShallowConvNet | None          | 65.74       | 54.30    | 96.13       | 60.70    | 87.05    | 82.70  |
|                | RDWT          | 66.32       | 55.10    | 95.94       | 62.30    | 87.27    | 87.27  |
| EEGNet         | None          | 70.79       | 61.10    | 95.85       | 59.60    | 87.32    | 83.10  |
|                | RDWT          | 70.10       | 60.10    | 96.06       | 64.00    | 88.08    | 84.10  |

### **Note**   
> - The recomputed results for these datasets (including accuracy/kappa scores) are available in their respective dataset folders.  
> - *Unlike the previous table, the results reported here for the HGD and BCI IV-2b datasets include an enhanced preprocessing pipeline, which incorporates data augmentation and class balancing techniques. These strategies were employed to address class imbalance and improve the generalization capabilities of the models.*
> - These values were obtained using our implementation and preprocessing pipeline. Minor deviations from the original papers are expected.

----

# None vs STFT vs RDWT in RatioWaveNet

| Dataset                  | Preprocessing | Accuracy (%) | Kappa (κ) |
|--------------------------|---------------|--------------|-----------|
| **BCI Competition IV-2a**| None          | 79.36        | 72.50     |
|                          | STFT          | 79.09        | 72.10     |
|                          | RDWT          | 81.56        | 75.40     |
| **BCI Competition IV-2b**| None          | 97.00        | 69.80     |
|                          | STFT          | 97.23        | 67.60     |
|                          | RDWT          | 97.69        | 80.60     |
| **High-Gamma Dataset**   | None          | 87.45        | 83.30     |
|                          | STFT          | 89.20        | 85.00     |
|                          | RDWT          | 92.81        | 90.40     |

---

# Dataset

This project uses three publicly available EEG motor imagery datasets for training and evaluation:

### 1. BCI Competition IV – Dataset 2a

- **Description**: EEG data from 9 subjects performing four different motor imagery tasks: left hand, right hand, feet, and tongue movements. Each subject completed two sessions (training and evaluation), with 288 trials per session.
- **Format**: `.mat` files
- **Download**: [BCI Competition IV – Dataset 2a](https://bnci-horizon-2020.eu/database/data-sets/001-2014/)

### 2. BCI Competition IV – Dataset 2b

- **Description**: EEG recordings from 9 subjects performing left and right hand motor imagery tasks. The dataset contains five sessions per subject, with three sessions including feedback.
- **Format**: `.gdf` files
- **Download**: [BCI Competition IV – Dataset 2b](https://www.bbci.de/competition/iv/download/)

### 3. High-Gamma Dataset (HGD)

- **Description**: EEG recordings from 14 subjects performing motor execution tasks, recorded using 128 channels. This dataset is well-suited for high-frequency EEG analysis.
- **Format**: `.mat` files
- **Download**: Available through the [GIN Repository](https://web.gin.g-node.org/robintibor/high-gamma-dataset)

> **Note**: Each dataset has a dedicated notebook in this repository, which includes download links and preprocessing instructions.


