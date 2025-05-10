## Risultati sui dataset EEG

| Model           | #params | BCI 4-2a Accuracy | BCI 4-2a Kappa | BCI 4-2b Accuracy | BCI 4-2b Kappa | HGD Accuracy | HGD Kappa |
|-----------------|---------|-------------------|----------------|-------------------|----------------|--------------|-----------|
| RockNetA        | 113,732 | 81.56             | 75.40          | 97.69             | 80.60          | 92.81        | 90.40     |
| ATCNet          | 113,732 | 81.10             | 79.73          | 89.41             | 78.80          | 92.05        | 89.40     |
| EEGTCNet        | 4,096   | 77.97             | 70.63          | 83.69             | 67.31          | 87.80        | 83.73     |
| MBEEG_SENet     | 10,170  | 79.98             | 73.30          | 86.53             | 73.02          | 90.13        | 86.84     |
| ShallowConvNet  | 47,310  | 80.52             | 74.02          | 86.02             | 72.38          | 87.00        | 82.67     |
| EEGNet          | 2,548   | 77.68             | 70.24          | 86.08             | 72.13          | 88.25        | 84.33     |

> **Note**: *HGD = High Gamma Dataset*

| Model           | Preprocessing | BCI 2a Acc. | BCI 2a κ | BCI 2b Acc. | BCI 2b κ | HGD Acc. | HGD κ |
|----------------|---------------|-------------|----------|-------------|----------|----------|--------|
| RockNetA       | None          |      
|                | RDWT          | 81.56       | 75.40    | 97.69       | 80.60    | 92.81    | 90.40  |
| ATCNet         | None          | 79.71       | 72.90    | 96.90       | 63.30    |
|                | RDWT          | 79.51       | 72.70    | 96.74       | 61.90    |
| EEGTCNet       | None          | 64.35       | 52.50    | 95.81       | 58.90    |
|                | RDWT          | 68.79       | 58.40    | 96.09       | 66.60    |
| MBEEG_SENet    | None          | 70.49       | 60.60    |
|                | RDWT          | 72.72       | 63.60    |
| ShallowConvNet | None          | 65.74       | 54.30    | 96.13       | 60.70    |
|                | RDWT          | 66.32       | 55.10    | 95.94       | 62.30    |
| EEGNet         | None          | 70.79       | 61.10    | 
|                | RDWT          | 70.10       | 60.10    |

> **Note**: *Unlike the previous table, the results reported here for the HGD and BCI IV-2b datasets include an enhanced preprocessing pipeline, which incorporates data augmentation and class balancing techniques. These strategies were employed to address class imbalance and improve the generalization capabilities of the models.*
